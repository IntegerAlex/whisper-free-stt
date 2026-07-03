# Calibration Logic

## Overview

At startup, STT samples the microphone for **1.5 seconds** to measure the ambient
noise floor. This measurement seeds the adaptive VAD with an initial noise estimate,
so the detector works correctly regardless of microphone gain, room acoustics, or
background hum.

The calibration runs **in parallel** with ASR model warm-up (loading the Whisper
model into memory), so there is no added startup latency.

## Algorithm

### Step 1: Collect ambient samples

```python
calib_rms: list[float] = []
stream_iter = mic_stream(config.audio)   # opens mic, starts streaming
deadline = time.monotonic() + 1.5        # 1.5 second window

while time.monotonic() < deadline:
    chunk = next(stream_iter)             # 1024 samples (64ms) per chunk
    ring.extend(chunk)                    # preserve chunks — nothing lost
    calib_rms.append(compute_rms(chunk)) # compute RMS for each chunk
```

At 16kHz with 1024-sample blocks, this produces approximately:
```
1.5s × (16000 / 1024) ≈ 23 chunks
```

### Step 2: Compute the 10th percentile

```python
sorted_rms = sorted(calib_rms)           # ascending order
p10 = sorted_rms[len(sorted_rms) // 10]  # value at index ~2
```

**Why the 10th percentile?** The median or mean would be skewed upward if the
user speaks or makes noise during calibration. The 10th percentile is a robust
estimate of the quietest sustained noise floor — it's resistant to transient
sounds (speech, chair creaks, keyboard) that briefly spike RMS.

Example with 23 chunks, sorted:
```
[0.002, 0.003, 0.003, 0.003, 0.003, 0.004, 0.004, 0.004, 0.004, 0.004,
 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.006, 0.006, 0.007, 0.008,
 0.012, 0.015, 0.450]    ← 0.450 was a chair creak during calibration

p10 = sorted[23 // 10] = sorted[2] = 0.003
```

The spike at 0.450 does not affect p10. The median (sorted[11] = 0.005) would
have been pulled up, and the mean even more so.

### Step 3: Seed the detector

```python
detector.set_noise_floor(p10)
```

This sets `detector._noise_floor = max(0.003, 1e-6) = 0.003`.

### Step 4: Compute initial thresholds

The detector computes adaptive thresholds from the noise floor using SNR-based
formulas (not fixed RMS thresholds):

```python
# End threshold: noise floor × SNR ratio
end_threshold = noise_floor × 10^(speech_threshold_db / 20)
             = 0.003 × 10^(6.0 / 20)
             = 0.003 × 1.995
             = 0.006

# Start threshold: end threshold + hysteresis margin
start_threshold = noise_floor × 10^((speech_threshold_db + hysteresis_up_db) / 20)
               = 0.003 × 10^(10.0 / 20)
               = 0.003 × 3.162
               = 0.0095
```

So after calibration with a quiet room (p10=0.003):
- Speech must exceed **RMS 0.0095** to trigger detection
- Speech drops below **RMS 0.006** to end the utterance
- The **hysteresis gap** is 0.0035 (start − end)

### Step 5: Spectral baselines (optional)

When `use_spectral_vad=True`, calibration also computes baselines for spectral
features (centroid, flux, ZCR, BER) from the ambient noise. These baselines
are used to normalize spectral features during runtime scoring.

## Ongoing Adaptation

After calibration, the noise floor continues to track via **dual-timescale EMA**
during non-speech periods:

### Slow EMA (30-minute baseline)
```python
noise_slow = 0.9999 * noise_slow + 0.0001 * min_energy
```

### Fast EMA (2-second rapid changes)
```python
noise_fast = 0.995 * noise_fast + 0.005 * min_energy
```

### Adaptive Blending
```python
# 10th percentile of 3-second energy history (robust to speech)
min_energy = percentile(energy_history, 10)

# Fast window detects rapid noise changes
if len(energy_history) >= 50:
    fast_min = percentile(last_50, 10)
    if abs(fast_min - noise_floor) > 0.02:
        # Rapid change: blend 50/50, faster alpha
        noise_floor = 0.5 * noise_floor + 0.5 * fast_min
        alpha = 0.99
    else:
        # Decay back to slow adaptation
        alpha = min(0.9999, alpha + 0.001)

noise_floor = alpha * noise_floor + (1 - alpha) * min_energy
noise_floor = clip(noise_floor, 0.001, 0.5)
```

The noise floor is also continuously updated by `update_imcra_noise()` which
tracks local minima of the power spectrum (simplified IMCRA).

## Why This Works

| Problem | Solution |
|---|---|
| Different mics have different gains | Calibration measures the actual noise level of the selected mic |
| User speaks during calibration | 10th percentile ignores transient speech spikes |
| Room noise changes over time | Dual-timescale EMA continues tracking after calibration |
| Silent room → threshold too low → false triggers | `max(..., 0.001)` floor prevents degenerate thresholds |
| Noisy room → threshold too high → miss speech | Caps at 0.5 prevent unusable thresholds |
| First utterance cold start | ASR model loaded in parallel thread during calibration |
| Non-stationary noise (fan on/off) | Fast EMA (2s) detects changes, slow EMA (30min) provides baseline |

## Edge Cases

### Dead-silent room (p10 ≈ 0.0)
```
p10 = 0.001
noise_floor = 0.001
end_th = 0.001 × 10^(6/20) = 0.002
start_th = 0.001 × 10^(10/20) = 0.003
```

### Noisy room (p10 = 0.05)
```
p10 = 0.05
noise_floor = 0.05
end_th = 0.05 × 10^(6/20) = 0.10
start_th = 0.05 × 10^(10/20) = 0.16
```

### All mics silent (calibration fails)
If no calibration data is collected (stream error), the detector starts with the
constructor default:
```
noise_floor = 0.005 / max(3.0, 1.0) = 0.00167
```
This is deliberately low, so the first few utterances will use conservative
thresholds that adapt upward quickly via EMA.

### Calibration overlap with warm-up
The ASR model warm-up (`warm_up_backend`) is launched as a daemon thread before
calibration begins. It runs completely in parallel — the mic stream and calibration
are on the main thread, while model loading happens on the warm-up thread. By the
time the first utterance is transcribed, the model is already in memory.

## Comparison with Full IMCRA

The calibration uses a simplified noise estimation. The full `StreamingEndpointDetector`
uses:

| Feature | Calibration | Runtime |
|---|---|---|
| Noise estimate | 10th percentile | Dual-timescale EMA + simplified IMCRA |
| Thresholds | Fixed SNR-based | SNR-based with adaptive margins |
| Spectral features | Baseline collection | Multi-feature fusion (flux, centroid, ZCR, BER) |
| Hangover | None | 150ms prevents truncation |
| State machine | None | SILENCE → SPEECH with hysteresis |
