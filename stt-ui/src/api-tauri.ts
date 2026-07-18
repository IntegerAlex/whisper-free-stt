// ── Tauri sidecar: spawns `stt-engine` via shell plugin ──
import { type STTApi, type STTEvent } from "./api";
import type { Child } from "@tauri-apps/plugin-shell";

let webAudioCapture: {
  stream: MediaStream;
  audioContext: AudioContext;
  source: MediaStreamAudioSourceNode;
  analyser: AnalyserNode;
} | null = null;

export async function requestWebAudioCapture(): Promise<boolean> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    webAudioCapture = { stream, audioContext, source, analyser };
    return true;
  } catch (err) {
    console.error("Failed to capture web audio", err);
    return false;
  }
}

export function stopWebAudioCapture(): void {
  if (webAudioCapture) {
    webAudioCapture.stream.getTracks().forEach((track) => track.stop());
    webAudioCapture.audioContext.close();
    webAudioCapture = null;
  }
}

export function getWebAudioAnalyser(): AnalyserNode | null {
  return webAudioCapture?.analyser ?? null;
}

export function createTauriApi(args: string[], sidecarName: string = "binaries/stt-engine"): STTApi {
  let listeners: Array<(e: STTEvent) => void> = [];
  let child: Child | null = null;
  // IMPORTANT: stdout and stderr are two independent, asynchronously-arriving
  // streams — stdout carries the line-delimited JSON event protocol
  // (`_json_emit` in the Python backend), stderr carries plain-text log
  // lines from the `logging` module. They MUST use separate buffers. A
  // previous version of this code buffered both streams through a single
  // shared `lineBuffer`, so if a stderr "data" event fired in between two
  // stdout chunks (or vice versa), fragments from the two unrelated streams
  // got concatenated together before the next newline — corrupting the
  // JSON line so it silently failed to parse and the event was dropped
  // (with only a garbled fallback console.log/warn, no error surfaced).
  // This caused state/transcript events (e.g. "transcribing", the final
  // transcript, or "idle") to intermittently vanish depending on exact
  // timing — the exact kind of "works most of the time, randomly fails"
  // behavior reported for PTT.
  let stdoutBuffer = "";
  let stderrBuffer = "";

  const notifyError = (msg: string) => {
    for (const cb of listeners) cb({ type: "error", message: msg });
  };

  const handleLine = (source: "stdout" | "stderr", data: string) => {
    if (source === "stdout") {
      stdoutBuffer += data;
      const lines = stdoutBuffer.split("\n");
      // Keep the last (potentially incomplete) chunk in the buffer
      stdoutBuffer = lines.pop() ?? "";
      for (const raw of lines) {
        const trimmed = raw.trim();
        if (!trimmed) continue;
        try {
          const event: STTEvent = JSON.parse(trimmed);
          for (const cb of listeners) cb(event);
        } catch {
          console.log(`[Sidecar stdout] ${trimmed}`);
        }
      }
    } else {
      stderrBuffer += data;
      const lines = stderrBuffer.split("\n");
      stderrBuffer = lines.pop() ?? "";
      for (const raw of lines) {
        const trimmed = raw.trim();
        if (!trimmed) continue;
        // stderr is plain-text logging, never JSON — no parse attempt needed.
        console.warn(`[Sidecar stderr] ${trimmed}`);
      }
    }
  };

  return {
    onEvent(cb) { listeners.push(cb); },

    async spawn() {
      const { Command } = await import("@tauri-apps/plugin-shell");
      const cmd = Command.sidecar(sidecarName, args);
      cmd.stdout.on("data", (line: string) => handleLine("stdout", line));
      cmd.stderr.on("data", (line: string) => handleLine("stderr", line));
      cmd.on("close", () => {
        for (const cb of listeners) cb({ type: "state", state: "idle" });
      });
      cmd.on("error", (err: string) => {
        notifyError(`Sidecar error: ${err}`);
      });
      child = await cmd.spawn();
      console.log("[Engine] Sidecar spawned — models loading, engine warming...");
    },

    kill() {
      listeners = [];
      if (child) {
        try { child.kill(); } catch { }
        child = null;
        console.log("[Engine] Sidecar killed");
      }
    },

    start() {
      this.sendCommand({ type: "start_recording" });
      console.log("[PTT] Sent start_recording");
    },

    stop() {
      this.sendCommand({ type: "stop_recording" });
      console.log("[PTT] Sent stop_recording");
    },

    sendCommand(cmd: Record<string, unknown>) {
      if (child) {
        child.write(JSON.stringify(cmd) + "\n").catch((e) => {
          console.warn("[Engine] Failed to write command:", e);
        });
      }
    },
  };
}
