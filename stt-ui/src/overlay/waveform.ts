/**
 * Real RMS waveform — drives SVG bars from live microphone audio via
 * AudioContext + AnalyserNode + FFT + requestAnimationFrame.
 *
 * Asymmetric smoothing: rise=0.55 (fast), fall=0.22 (slow) makes the
 * bars feel alive and responsive.
 */

export interface WaveformConfig {
  barCount: number;
  fftSize?: number;          // Default 64 (32 frequency bins)
  minDecibels?: number;      // Default -90
  maxDecibels?: number;      // Default -10
  smoothingTimeConstant?: number; // Default 0.4
  riseSmoothing?: number;    // Default 0.55
  fallSmoothing?: number;    // Default 0.22
  barWidth?: number;         // Default 3
  barGap?: number;           // Default 2
  barRadius?: number;        // Default 1.5
}

interface WaveformState {
  audioContext: AudioContext | null;
  analyser: AnalyserNode | null;
  stream: MediaStream | null;
  dataArray: Uint8Array | null;
  smoothed: Float32Array | null;
  rafId: number | null;
  onBars: ((bars: number[]) => void) | null;
  active: boolean;
}

export function createWaveform(config: WaveformConfig) {
  const {
    barCount,
    fftSize = 64,
    minDecibels = -90,
    maxDecibels = -10,
    smoothingTimeConstant = 0.4,
    riseSmoothing = 0.55,
    fallSmoothing = 0.22,
  } = config;

  const state: WaveformState = {
    audioContext: null,
    analyser: null,
    stream: null,
    dataArray: null,
    smoothed: null,
    rafId: null,
    onBars: null,
    active: false,
  };

  function computeBars(): number[] {
    if (!state.analyser || !state.dataArray || !state.smoothed) {
      return new Array(barCount).fill(0);
    }

    state.analyser.getByteFrequencyData(state.dataArray);

    // Map FFT bins to bars (downsample if more bins than bars)
    const binsPerBar = Math.floor(state.dataArray.length / barCount);
    const bars: number[] = [];

    for (let i = 0; i < barCount; i++) {
      // Average the frequency bins for this bar
      let sum = 0;
      const start = i * binsPerBar;
      for (let j = start; j < start + binsPerBar && j < state.dataArray.length; j++) {
        sum += state.dataArray[j];
      }
      const raw = sum / binsPerBar / 255; // Normalize to 0-1

      // Asymmetric smoothing — fast rise, slow fall
      const prev = state.smoothed[i];
      const smoothing = raw > prev ? riseSmoothing : fallSmoothing;
      const smoothed = prev + (raw - prev) * smoothing;
      state.smoothed[i] = smoothed;

      bars.push(smoothed);
    }

    return bars;
  }

  function tick() {
    if (!state.active) return;
    const bars = computeBars();
    state.onBars?.(bars);
    state.rafId = requestAnimationFrame(tick);
  }

  return {
    /** Start capturing audio and emitting bar values. */
    async start() {
      if (state.active) return;

      try {
        // Create AudioContext
        state.audioContext = new AudioContext();
        state.analyser = state.audioContext.createAnalyser();
        state.analyser.fftSize = fftSize;
        state.analyser.minDecibels = minDecibels;
        state.analyser.maxDecibels = maxDecibels;
        state.analyser.smoothingTimeConstant = smoothingTimeConstant;

        // Get microphone stream
        state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Connect: mic → analyser
        const source = state.audioContext.createMediaStreamSource(state.stream);
        source.connect(state.analyser);

        // Allocate buffers
        state.dataArray = new Uint8Array(state.analyser.frequencyBinCount);
        state.smoothed = new Float32Array(barCount);

        state.active = true;
        state.rafId = requestAnimationFrame(tick);
      } catch (err) {
        console.error("[waveform] Failed to start:", err);
        // Fallback: start fake waveform
        state.active = true;
        state.smoothed = new Float32Array(barCount);
        state.rafId = requestAnimationFrame(tick);
      }
    },

    /** Stop capturing audio. */
    stop() {
      state.active = false;
      if (state.rafId !== null) {
        cancelAnimationFrame(state.rafId);
        state.rafId = null;
      }
      if (state.stream) {
        state.stream.getTracks().forEach((t) => t.stop());
        state.stream = null;
      }
      if (state.audioContext) {
        state.audioContext.close().catch(() => {});
        state.audioContext = null;
      }
      state.analyser = null;
      state.dataArray = null;
      state.smoothed = null;
    },

    /** Set the callback that receives bar values. */
    onBars(cb: (bars: number[]) => void) {
      state.onBars = cb;
    },

    /** Check if actively capturing. */
    isActive() {
      return state.active;
    },
  };
}

/**
 * SVG bar renderer for the waveform.
 * Returns a function that updates an SVG element's bar heights.
 */
export function createWaveformRenderer(
  svgEl: SVGSVGElement,
  config: WaveformConfig
) {
  const { barCount, barWidth = 3, barGap = 2, barRadius = 1.5 } = config;
  const totalWidth = barCount * (barWidth + barGap) - barGap;
  const maxHeight = 24;

  // Create bar elements
  const bars: SVGRectElement[] = [];
  for (let i = 0; i < barCount; i++) {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(i * (barWidth + barGap)));
    rect.setAttribute("width", String(barWidth));
    rect.setAttribute("rx", String(barRadius));
    rect.setAttribute("ry", String(barRadius));
    rect.setAttribute("fill", "currentColor");
    svgEl.appendChild(rect);
    bars.push(rect);
  }

  svgEl.setAttribute("viewBox", `0 0 ${totalWidth} ${maxHeight}`);

  return function update(values: number[]) {
    for (let i = 0; i < bars.length; i++) {
      const v = values[i] ?? 0;
      const h = Math.max(2, v * maxHeight); // Minimum height of 2px
      const y = (maxHeight - h) / 2;
      bars[i].setAttribute("y", String(y));
      bars[i].setAttribute("height", String(h));
    }
  };
}
