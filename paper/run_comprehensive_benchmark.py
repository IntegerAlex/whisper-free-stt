#!/usr/bin/env python3
"""Comprehensive benchmark suite for the Floure STT paper.

Part 1: WER on LibriSpeech test-clean
Part 2: VAD Ablation Study
Part 3: Latency percentiles (P50/P95/P99) on 50 utterances
"""
import time
import json
import sys
import os
import statistics
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))


def make_speech_burst(sr: int, duration: float, freq: float = 200.0, amplitude: float = 0.3) -> np.ndarray:
    """Generate a speech-like burst with harmonic structure."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    signal += (amplitude * 0.5) * np.sin(2 * np.pi * freq * 2 * t)
    signal += (amplitude * 0.25) * np.sin(2 * np.pi * freq * 3 * t)
    # AM envelope (syllable-like)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t))
    signal *= envelope
    return signal.astype(np.float32)


def add_noise(signal: np.ndarray, snr_db: float, sr: int) -> np.ndarray:
    """Add white noise at a target SNR."""
    sig_power = np.mean(signal ** 2)
    if sig_power < 1e-10:
        return signal
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(signal)).astype(np.float32) * np.sqrt(noise_power)
    return (signal + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Part 1: WER Benchmark
# ---------------------------------------------------------------------------

def run_wer_benchmark(output_dir: str = "/tmp/librispeech_test_clean") -> dict:
    """Download LibriSpeech test-clean and compute WER across profiles."""
    print("\n" + "=" * 60)
    print("PART 1: WER ON LIBRISPEECH TEST-CLEAN")
    print("=" * 60)

    # Step 1: Try loading dataset
    ds = None
    try:
        from datasets import load_dataset
        print("[1/4] Loading LibriSpeech test-clean...")
        ds = load_dataset(
            "openslr/librispeech_asr",
            "test.clean",
            split="validation",
            trust_remote_code=True,
            streaming=True,
        )
        print("  Dataset loaded (streaming mode).")
    except Exception as e:
        print(f"  Failed to load dataset: {e}")
        print("  Falling back to synthetic benchmark.")
        return run_synthetic_wer_benchmark()

    # Step 2: Collect 50 samples with text
    print("[2/4] Collecting 50 utterances...")
    samples = []
    for i, item in enumerate(ds):
        if len(samples) >= 50:
            break
        audio = item["audio"]["array"]
        text = item["text"].strip()
        if len(text) > 10 and len(audio) > 0.5 * 16000:
            samples.append({
                "audio": np.array(audio, dtype=np.float32),
                "text": text,
                "sr": item["audio"]["sampling_rate"],
            })
    print(f"  Collected {len(samples)} samples.")

    if len(samples) < 10:
        print("  Too few samples. Falling back to synthetic benchmark.")
        return run_synthetic_wer_benchmark()

    # Step 3: Install jiwer
    print("[3/4] Installing jiwer...")
    os.system(f"{sys.executable} -m pip install jiwer -q 2>/dev/null")

    from jiwer import wer as compute_wer_metric

    # Step 4: Run transcription across profiles
    from stt.config import TranscriptionConfig, TranscriptionBackend
    from stt.transcription import transcribe

    profiles = [
        ("tiny.en", TranscriptionBackend.WHISPER_CPP, "tiny.en", 1),
        ("base.en", TranscriptionBackend.WHISPER_CPP, "base.en", 1),
        ("small.en", TranscriptionBackend.WHISPER_CPP, "small.en", 3),
    ]

    # Add turbo if CUDA available
    try:
        import torch
        if torch.cuda.is_available():
            profiles.append(("large-v3-turbo", TranscriptionBackend.FASTER_WHISPER, "large-v3-turbo", 5))
    except ImportError:
        pass

    results = {"profiles": {}}
    print("[4/4] Running WER evaluation...")

    for profile_name, backend, model_name, beam_size in profiles:
        print(f"\n  --- {profile_name} ({backend.value}, beam={beam_size}) ---")
        config = TranscriptionConfig(
            backend=backend,
            model_name=model_name,
            beam_size=beam_size,
            noise_reduce=True,
            vad_filter=(backend == TranscriptionBackend.FASTER_WHISPER),
        )

        hypotheses = []
        references = []
        latencies = []

        for idx, sample in enumerate(samples):
            t0 = time.perf_counter()
            try:
                result = transcribe(sample["audio"], sample["sr"], config)
                t1 = time.perf_counter()
                hyp = result.text.strip() if result.text else ""
                ref = sample["text"].strip()
                if hyp and ref:
                    hypotheses.append(hyp)
                    references.append(ref)
                    latencies.append(t1 - t0)
                    if (idx + 1) % 10 == 0:
                        print(f"    [{idx+1}/{len(samples)}] latency={t1-t0:.3f}s  hyp={hyp[:50]}...")
            except Exception as e:
                print(f"    [{idx+1}/{len(samples)}] ERROR: {e}")

        if hypotheses:
            wer_score = compute_wer_metric(references, hypotheses)
            wer_pct = round(wer_score * 100, 2)
            results["profiles"][profile_name] = {
                "n": len(hypotheses),
                "wer_pct": wer_pct,
                "median_latency_s": round(statistics.median(latencies), 3),
                "mean_latency_s": round(statistics.mean(latencies), 3),
                "std_latency_s": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0.0,
                "backend": backend.value,
                "model": model_name,
            }
            print(f"    WER = {wer_pct}%  (n={len(hypotheses)}, "
                  f"median={results['profiles'][profile_name]['median_latency_s']}s)")
        else:
            print(f"    No valid transcriptions for {profile_name}")
            results["profiles"][profile_name] = {"n": 0, "wer_pct": None, "error": "no valid transcriptions"}

    return results


def run_synthetic_wer_benchmark() -> dict:
    """Fallback: synthetic benchmark with known text."""
    print("\n  Running SYNTHETIC WER benchmark...")
    from stt.config import TranscriptionConfig, TranscriptionBackend
    from stt.transcription import transcribe

    # Known phrases and their expected transcriptions
    phrases = [
        ("the quick brown fox jumps over the lazy dog", 180.0),
        ("hello world this is a test", 220.0),
        ("speech recognition is very important", 190.0),
        ("the weather is nice today", 210.0),
        ("machine learning models are improving rapidly", 200.0),
        ("open source software drives innovation", 170.0),
        "the cat sat on the mat",
        "good morning how are you today",
        "artificial intelligence will transform the world",
        "please call the doctor tomorrow morning",
    ] * 5  # 50 samples

    sr = 16000
    samples = []
    for item in phrases:
        if isinstance(item, tuple):
            text, freq = item
        else:
            text, freq = item, 200.0
        duration = max(1.5, min(4.0, len(text) * 0.07))
        audio = make_speech_burst(sr, duration, freq=freq, amplitude=0.3)
        samples.append({"audio": audio, "text": text, "sr": sr})

    try:
        from jiwer import wer as compute_wer_metric
    except ImportError:
        os.system(f"{sys.executable} -m pip install jiwer -q")
        from jiwer import wer as compute_wer_metric

    profiles = [
        ("tiny.en", TranscriptionBackend.WHISPER_CPP, "tiny.en", 1),
        ("base.en", TranscriptionBackend.WHISPER_CPP, "base.en", 1),
        ("small.en", TranscriptionBackend.WHISPER_CPP, "small.en", 3),
    ]

    results = {"profiles": {}, "synthetic": True}

    for profile_name, backend, model_name, beam_size in profiles:
        print(f"\n  --- {profile_name} ---")
        config = TranscriptionConfig(
            backend=backend,
            model_name=model_name,
            beam_size=beam_size,
            noise_reduce=True,
        )

        hypotheses = []
        references = []
        latencies = []

        for idx, sample in enumerate(samples):
            t0 = time.perf_counter()
            try:
                result = transcribe(sample["audio"], sample["sr"], config)
                t1 = time.perf_counter()
                hyp = result.text.strip() if result.text else ""
                ref = sample["text"].strip()
                if hyp:
                    hypotheses.append(hyp)
                    references.append(ref)
                    latencies.append(t1 - t0)
            except Exception as e:
                if idx < 3:
                    print(f"    ERROR: {e}")

        if hypotheses:
            wer_score = compute_wer_metric(references, hypotheses)
            results["profiles"][profile_name] = {
                "n": len(hypotheses),
                "wer_pct": round(wer_score * 100, 2),
                "median_latency_s": round(statistics.median(latencies), 3),
                "mean_latency_s": round(statistics.mean(latencies), 3),
            }
            print(f"    WER = {results['profiles'][profile_name]['wer_pct']}%")

    return results


# ---------------------------------------------------------------------------
# Part 2: VAD Ablation Study
# ---------------------------------------------------------------------------

def run_vad_ablation() -> dict:
    """VAD ablation: energy-only, spectral-only, full composite, Silero."""
    print("\n" + "=" * 60)
    print("PART 2: VAD ABLATION STUDY")
    print("=" * 60)

    from stt.vad import StreamingEndpointDetector, VADState
    from stt.config import VADConfig

    sr = 16000
    block_size = 1024
    blocks_per_sec = sr // block_size  # ~15.6

    vad_configs = {
        "energy_only": VADConfig(use_spectral_vad=True, spectral_weight=0.0),
        "spectral_only": VADConfig(use_spectral_vad=True, spectral_weight=1.0),
        "full": VADConfig(use_spectral_vad=True, spectral_weight=0.4),
    }

    # Try Silero
    silero_available = False
    try:
        from silero_vad import get_speech_timestamps, load_speech_model
        silero_model = load_speech_model()
        silero_available = True
        print("  Silero VAD available.")
    except Exception as e:
        print(f"  Silero VAD not available: {e}")

    # Test conditions: (name, snr_db, duration_sec)
    conditions = [
        ("clean_speech", 20.0, 30.0),
        ("noisy_speech", 10.0, 30.0),
        ("very_noisy", 5.0, 30.0),
    ]

    results = {}

    for vad_name, vad_cfg in vad_configs.items():
        print(f"\n  --- VAD: {vad_name} ---")
        results[vad_name] = {}

        for cond_name, snr_db, duration in conditions:
            detector = StreamingEndpointDetector(vad_cfg, sr, block_size)
            num_blocks = int(duration * sr / block_size)

            speech_events = []  # (kind, block_idx)
            speech_intervals = []  # ground truth intervals
            block_times = np.arange(num_blocks) * block_size / sr

            # Generate ground truth: speech every 2 seconds for 0.5s
            gt_speech = np.zeros(num_blocks, dtype=bool)
            for b in range(num_blocks):
                t = block_times[b]
                cycle_pos = t % 2.0
                if cycle_pos < 0.5:
                    gt_speech[b] = True

            # Run VAD
            for i in range(num_blocks):
                t = block_times[i]
                cycle_pos = t % 2.0
                if cycle_pos < 0.5:
                    signal = make_speech_burst(sr, block_size / sr, freq=200.0, amplitude=0.3)
                else:
                    signal = np.zeros(block_size, dtype=np.float32)

                noise = np.random.randn(block_size).astype(np.float32) * 0.01
                chunk = add_noise(signal, snr_db, sr)

                r = rms(chunk)
                chunk_start = i * block_size
                chunk_end = chunk_start + block_size

                event = detector.update(r, chunk_start, chunk_end, chunk)
                if event:
                    speech_events.append((event.kind, i))

            # Compute metrics
            detected_blocks = np.zeros(num_blocks, dtype=bool)
            for kind, blk in speech_events:
                if kind == "start":
                    # Mark a window as detected
                    end_blk = min(num_blocks, blk + int(0.5 * blocks_per_sec))
                    detected_blocks[blk:end_blk] = True

            # Detection rate: fraction of GT speech blocks that were detected
            gt_speech_count = np.sum(gt_speech)
            if gt_speech_count > 0:
                true_positives = np.sum(gt_speech & detected_blocks)
                detection_rate = true_positives / gt_speech_count
            else:
                detection_rate = 0.0

            # False alarm rate: fraction of GT silence blocks that were incorrectly detected
            gt_silence = ~gt_speech
            gt_silence_count = np.sum(gt_silence)
            if gt_silence_count > 0:
                false_positives = np.sum(gt_silence & detected_blocks)
                false_alarm_rate = false_positives / gt_silence_count
            else:
                false_alarm_rate = 0.0

            # Detection latency: time from first GT speech block to first detection
            gt_speech_indices = np.where(gt_speech)[0]
            detected_indices = np.where(detected_blocks)[0]
            if len(gt_speech_indices) > 0 and len(detected_indices) > 0:
                first_gt = gt_speech_indices[0]
                first_det = detected_indices[0]
                detection_latency = max(0.0, (first_det - first_gt) * block_size / sr)
            else:
                detection_latency = float('inf') if gt_speech_count > 0 else 0.0

            results[vad_name][cond_name] = {
                "detection_rate": round(float(detection_rate), 4),
                "false_alarm_rate": round(float(false_alarm_rate), 4),
                "detection_latency_ms": round(float(detection_latency) * 1000, 1),
                "snr_db": snr_db,
                "duration_s": duration,
            }
            print(f"    {cond_name} (SNR={snr_db}dB): "
                  f"det={detection_rate:.1%}  fa={false_alarm_rate:.1%}  "
                  f"lat={detection_latency*1000:.0f}ms")

    # Silero ablation
    if silero_available:
        print(f"\n  --- VAD: silero ---")
        results["silero"] = {}
        for cond_name, snr_db, duration in conditions:
            num_blocks = int(duration * sr / block_size)
            block_times = np.arange(num_blocks) * block_size / sr

            # Build full audio
            full_audio = np.zeros(num_blocks * block_size, dtype=np.float32)
            gt_speech = np.zeros(num_blocks, dtype=bool)
            for i in range(num_blocks):
                t = block_times[i]
                cycle_pos = t % 2.0
                start = i * block_size
                end = start + block_size
                if cycle_pos < 0.5:
                    sig = make_speech_burst(sr, block_size / sr, freq=200.0, amplitude=0.3)
                    full_audio[start:end] = add_noise(sig, snr_db, sr)
                    gt_speech[i] = True
                else:
                    full_audio[start:end] = np.random.randn(block_size).astype(np.float32) * 0.001

            try:
                speech_ts = get_speech_timestamps(
                    full_audio, silero_model,
                    sampling_rate=sr,
                    threshold=0.5,
                    min_speech_duration_ms=200,
                    min_silence_duration_ms=300,
                )
                # Convert Silero timestamps to block-level detection
                detected_blocks = np.zeros(num_blocks, dtype=bool)
                for ts in speech_ts:
                    start_block = ts["start"] // block_size
                    end_block = ts["end"] // block_size
                    detected_blocks[start_block:min(end_block + 1, num_blocks)] = True

                gt_speech_count = np.sum(gt_speech)
                gt_silence_count = np.sum(~gt_speech)
                detection_rate = float(np.sum(gt_speech & detected_blocks) / gt_speech_count) if gt_speech_count > 0 else 0.0
                false_alarm_rate = float(np.sum(~gt_speech & detected_blocks) / gt_silence_count) if gt_silence_count > 0 else 0.0

                gt_indices = np.where(gt_speech)[0]
                det_indices = np.where(detected_blocks)[0]
                if len(gt_indices) > 0 and len(det_indices) > 0:
                    detection_latency = max(0.0, (det_indices[0] - gt_indices[0]) * block_size / sr)
                else:
                    detection_latency = float('inf') if gt_speech_count > 0 else 0.0

                results["silero"][cond_name] = {
                    "detection_rate": round(detection_rate, 4),
                    "false_alarm_rate": round(false_alarm_rate, 4),
                    "detection_latency_ms": round(detection_latency * 1000, 1),
                    "snr_db": snr_db,
                    "duration_s": duration,
                }
                print(f"    {cond_name} (SNR={snr_db}dB): "
                      f"det={detection_rate:.1%}  fa={false_alarm_rate:.1%}  "
                      f"lat={detection_latency*1000:.0f}ms")
            except Exception as e:
                results["silero"][cond_name] = {"error": str(e)}
                print(f"    {cond_name}: Silero failed: {e}")
    else:
        results["silero"] = {"available": False, "error": "silero-vad not installed"}

    return results


# ---------------------------------------------------------------------------
# Part 3: Latency Percentiles (50 utterances)
# ---------------------------------------------------------------------------

def run_latency_large_sample() -> dict:
    """Run 50 utterances through turbo profile, report P50/P95/P99."""
    print("\n" + "=" * 60)
    print("PART 3: LATENCY PERCENTILES (50 UTTERANCES)")
    print("=" * 60)

    from stt.config import TranscriptionConfig, TranscriptionBackend
    from stt.transcription import transcribe

    # Try turbo, fall back to small.en on CPU
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if has_cuda:
        config = TranscriptionConfig(
            backend=TranscriptionBackend.FASTER_WHISPER,
            model_name="large-v3-turbo",
            beam_size=5,
            noise_reduce=True,
            vad_filter=True,
            batch_size=8,
        )
        profile_name = "large-v3-turbo"
    else:
        config = TranscriptionConfig(
            backend=TranscriptionBackend.WHISPER_CPP,
            model_name="small.en",
            beam_size=3,
            noise_reduce=True,
        )
        profile_name = "small.en (CPU fallback)"

    print(f"  Profile: {profile_name}")

    # Generate 50 diverse speech-like audio clips
    sr = 16000
    phrases = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a speech recognition test",
        "machine learning models are getting better every year",
        "open source software drives innovation in technology",
        "the weather today is absolutely beautiful and sunny",
        "artificial intelligence will change the way we work",
        "please schedule a meeting with the doctor for tomorrow",
        "the patient presents with acute onset chest pain",
        "differential diagnosis includes pneumonia and bronchitis",
        "we need to order a complete blood count and metabolic panel",
    ] * 5  # 50 samples

    latencies = []
    results_data = []

    for idx, phrase in enumerate(phrases):
        duration = max(1.5, min(4.0, len(phrase) * 0.07))
        freq = 180.0 + (idx % 5) * 20.0  # Vary pitch
        audio = make_speech_burst(sr, duration, freq=freq, amplitude=0.3)

        t0 = time.perf_counter()
        try:
            result = transcribe(audio, sr, config)
            t1 = time.perf_counter()
            latency = t1 - t0
            latencies.append(latency)
            results_data.append({
                "idx": idx,
                "latency_s": round(latency, 4),
                "text": result.text.strip()[:60] if result.text else "",
            })
            if (idx + 1) % 10 == 0:
                print(f"    [{idx+1}/50] latency={latency:.3f}s")
        except Exception as e:
            print(f"    [{idx+1}/50] ERROR: {e}")

    if not latencies:
        return {"n": 0, "error": "all transcriptions failed"}

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    result = {
        "n": n,
        "profile": profile_name,
        "p50": round(float(np.percentile(sorted_lat, 50)), 3),
        "p95": round(float(np.percentile(sorted_lat, 95)), 3),
        "p99": round(float(np.percentile(sorted_lat, 99)), 3),
        "mean": round(statistics.mean(sorted_lat), 3),
        "std": round(statistics.stdev(sorted_lat), 3) if n > 1 else 0.0,
        "min": round(sorted_lat[0], 3),
        "max": round(sorted_lat[-1], 3),
    }

    print(f"\n  Results (n={n}):")
    print(f"    P50 = {result['p50']}s")
    print(f"    P95 = {result['p95']}s")
    print(f"    P99 = {result['p99']}s")
    print(f"    Mean = {result['mean']}s  Std = {result['std']}s")
    print(f"    Range: [{result['min']}s, {result['max']}s]")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("FLOURE STT — COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    # Part 1: WER
    try:
        all_results["wer"] = run_wer_benchmark()
    except Exception as e:
        print(f"\nPart 1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["wer"] = {"error": str(e)}

    # Part 2: VAD Ablation
    try:
        all_results["vad_ablation"] = run_vad_ablation()
    except Exception as e:
        print(f"\nPart 2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["vad_ablation"] = {"error": str(e)}

    # Part 3: Latency
    try:
        all_results["latency_large"] = run_latency_large_sample()
    except Exception as e:
        print(f"\nPart 3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["latency_large"] = {"error": str(e)}

    # Metadata
    all_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_path}")
    print("=" * 60)

    # Print summary
    print("\nSUMMARY:")
    print("-" * 40)
    if "wer" in all_results and "profiles" in all_results.get("wer", {}):
        for pname, pdata in all_results["wer"]["profiles"].items():
            if pdata.get("wer_pct") is not None:
                print(f"  WER {pname}: {pdata['wer_pct']}% (n={pdata['n']})")
    if "latency_large" in all_results:
        lat = all_results["latency_large"]
        if lat.get("n", 0) > 0:
            print(f"  Latency {lat.get('profile','?')}: P50={lat['p50']}s P95={lat['p95']}s P99={lat['p99']}s")
    if "vad_ablation" in all_results:
        for vad_name, vad_data in all_results["vad_ablation"].items():
            if isinstance(vad_data, dict) and "error" not in vad_data:
                for cond, metrics in vad_data.items():
                    if isinstance(metrics, dict) and "detection_rate" in metrics:
                        print(f"  VAD {vad_name}/{cond}: det={metrics['detection_rate']:.0%} fa={metrics['false_alarm_rate']:.0%}")

    return all_results


if __name__ == "__main__":
    main()
