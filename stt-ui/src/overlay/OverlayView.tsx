import { useEffect, useRef, useState } from "react";
import { Mic } from "lucide-react";
import { createWaveform, type WaveformConfig } from "./waveform";
import { createSpring, SPRING_SNAPPY } from "./spring";

type OverlayState = "idle" | "listening" | "processing" | "inserting" | "success";

const WAVEFORM_CONFIG: WaveformConfig = {
  barCount: 14,
  fftSize: 64,
  riseSmoothing: 0.55,
  fallSmoothing: 0.22,
  barWidth: 3,
  barGap: 2,
  barRadius: 1.5,
};

function stateToLabel(state: OverlayState): string {
  switch (state) {
    case "idle": return "";
    case "listening": return "Listening...";
    case "processing": return "Transcribing...";
    case "inserting": return "Inserting...";
    case "success": return "Done";
    default: return "";
  }
}

function stateToColor(state: OverlayState): string {
  switch (state) {
    case "success": return "#22c55e";
    case "processing":
    case "inserting": return "#f59e0b";
    default: return "#FF3B56";
  }
}

export default function OverlayView() {
  const [state, setState] = useState<OverlayState>("idle");

  const svgRef = useRef<SVGSVGElement>(null);
  const waveformRef = useRef<ReturnType<typeof createWaveform> | null>(null);
  const rendererInitRef = useRef(false);
  const scaleSpring = createSpring(SPRING_SNAPPY);

  // Listen for overlay commands from the frontend (main window)
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let retries = 0;
    const MAX_RETRIES = 5;
    const RETRY_DELAY = 200;

    const tryListen = async (): Promise<boolean> => {
      try {
        const { listen } = await import("@tauri-apps/api/event");

        unlisten = await listen<OverlayState>("overlay:command", (event) => {
          const newState = event.payload;
          console.log(`[Overlay] Received command: ${newState}`);
          setState(newState);

          // Start waveform
          if (newState === "listening" && waveformRef.current && !waveformRef.current.isActive()) {
            console.log("[Overlay] Starting waveform");
            waveformRef.current.start();
          }

          // Stop waveform
          if (newState !== "listening" && waveformRef.current?.isActive()) {
            console.log("[Overlay] Stopping waveform");
            waveformRef.current.stop();
          }

          // Full cleanup — reset to initial state, then signal ready
          if (newState === "idle") {
            console.log("[Overlay] Cleaning up");
            waveformRef.current?.stop();
            scaleSpring.setValue(0);

            // Reset SVG bars to minimum height
            const svg = svgRef.current;
            if (svg) {
              const bars = svg.querySelectorAll("rect");
              bars.forEach((bar) => {
                bar.setAttribute("y", "11");
                bar.setAttribute("height", "2");
              });
            }

            rendererInitRef.current = false;
            console.log("[Overlay] Cleanup complete");

            // Acknowledge cleanup — App.tsx waits for this before hiding
            import("@tauri-apps/api/event").then(({ emitTo }) => {
              emitTo("main", "overlay:idle_ready", {});
              console.log("[Overlay] Sent idle_ready");
            }).catch(() => {});
          }
        });
        return true;
      } catch {
        return false;
      }
    };

    (async () => {
      while (retries < MAX_RETRIES) {
        if (await tryListen()) return;
        retries++;
        if (retries < MAX_RETRIES) {
          await new Promise((r) => setTimeout(r, RETRY_DELAY));
        }
      }
      console.warn("[Overlay] Failed to register listener after retries — staying idle");
    })();

    return () => {
      unlisten?.();
      waveformRef.current?.stop();
    };
  }, []);

  // Initialize waveform and SVG renderer
  useEffect(() => {
    if (svgRef.current && !rendererInitRef.current) {
      rendererInitRef.current = true;
      const svg = svgRef.current;
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      const bars: SVGRectElement[] = [];
      const { barCount, barWidth = 3, barGap = 2, barRadius = 1.5 } = WAVEFORM_CONFIG;
      const totalWidth = barCount * (barWidth + barGap) - barGap;
      const maxHeight = 24;

      for (let i = 0; i < barCount; i++) {
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", String(i * (barWidth + barGap)));
        rect.setAttribute("width", String(barWidth));
        rect.setAttribute("rx", String(barRadius));
        rect.setAttribute("ry", String(barRadius));
        rect.setAttribute("fill", "currentColor");
        rect.setAttribute("y", String(maxHeight / 2 - 1));
        rect.setAttribute("height", "2");
        svg.appendChild(rect);
        bars.push(rect);
      }
      svg.setAttribute("viewBox", `0 0 ${totalWidth} ${maxHeight}`);

      const wf = createWaveform(WAVEFORM_CONFIG);
      waveformRef.current = wf;

      wf.onBars((values) => {
        for (let i = 0; i < bars.length; i++) {
          const v = values[i] ?? 0;
          const h = Math.max(2, v * maxHeight);
          const y = (maxHeight - h) / 2;
          bars[i].setAttribute("y", String(y));
          bars[i].setAttribute("height", String(h));
        }
      });
    }
  }, []);

  // Spring animation for scale
  useEffect(() => {
    scaleSpring.onUpdate((v) => {
      const el = document.getElementById("overlay-pill");
      if (el) {
        el.style.transform = `scale(${0.92 + v * 0.08})`;
      }
    });
    scaleSpring.setValue(0);
  }, []);

  useEffect(() => {
    if (state !== "idle") {
      scaleSpring.setTarget(1);
    } else {
      scaleSpring.setTarget(0);
    }
  }, [state]);

  if (state === "idle") return null;

  const color = stateToColor(state);
  const label = stateToLabel(state);
  const showWaveform = state === "listening";
  const showCheckmark = state === "success";
  const showTranscribing = state === "processing" || state === "inserting";

  return (
    <div
      id="overlay-pill"
      className="fixed inset-0 flex items-center justify-center pointer-events-none select-none"
      style={{ opacity: 1 }}
    >
      <div
        className="flex items-center gap-3 px-4 py-2.5 rounded-full backdrop-blur-xl border shadow-lg"
        style={{
          backgroundColor: "rgba(30, 25, 22, 0.92)",
          borderColor: "rgba(255, 255, 255, 0.1)",
          boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
          minWidth: "200px",
        }}
      >
        {/* Mic icon with pulsing background */}
        <div className="relative flex items-center justify-center w-[28px] h-[28px]">
          <div
            className="absolute inset-0 rounded-full"
            style={{
              backgroundColor: color,
              opacity: showWaveform ? 0.3 : 0.15,
              animation: showWaveform ? "overlay-pulse 1.5s ease-in-out infinite" : "none",
            }}
          />
          {showCheckmark ? (
            <svg
              width="14"
              height="14"
              viewBox="0 0 14 14"
              fill="none"
              className="relative z-10"
            >
              <path
                d="M2.5 7L5.5 10L11.5 4"
                stroke="white"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          ) : (
            <Mic size={14} className="relative z-10 text-white" />
          )}
        </div>

        {/* Status label */}
        <span
          className="text-[13px] font-medium"
          style={{
            color: showTranscribing ? "rgba(255,255,255,0.6)" : "rgba(255,255,255,0.9)",
            transition: "color 150ms ease",
          }}
        >
          {label}
        </span>

        {/* Waveform bars */}
        {showWaveform && (
          <svg
            ref={svgRef}
            className="ml-1 text-white"
            width="60"
            height="24"
          />
        )}

        {/* Transcribing dots animation */}
        {showTranscribing && (
          <div className="flex items-center gap-[3px] ml-1">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="w-[3px] h-[3px] rounded-full bg-white/60"
                style={{
                  animation: `overlay-dot 1.2s ease-in-out ${i * 0.2}s infinite`,
                }}
              />
            ))}
          </div>
        )}
      </div>

      <style>{`
        @keyframes overlay-pulse {
          0%, 100% { opacity: 0.2; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.15); }
        }
        @keyframes overlay-dot {
          0%, 100% { opacity: 0.3; transform: translateY(0); }
          50% { opacity: 0.8; transform: translateY(-3px); }
        }
      `}</style>
    </div>
  );
}
