// ── Tauri native backend: uses Rust commands + Tauri events ──
import { type STTApi, type STTEvent } from "./api";

export function createTauriApi(): STTApi {
  let listeners: Array<(e: STTEvent) => void> = [];
  let unlistenFns: Array<() => void> = [];

  const emit = (e: STTEvent) => {
    for (const cb of listeners) cb(e);
  };

  const handleEvent = (name: string, payload: any) => {
    currentStatus = "listening";
    switch (name) {
      case "asr_ready":
        emit({ type: "asr_ready", backend: payload.backend ?? "parakeet" });
        emit({ type: "state", state: "idle" });
        break;
      case "asr_partial":
        currentStatus = "transcribing";
        emit({ type: "asr_partial", text: payload.text ?? "" });
        break;
      case "asr_final":
        emit({ type: "asr_final", text: payload.text ?? "", latency_ms: payload.latency_ms ?? 0 });
        break;
      case "llm_start":
        currentStatus = "rewriting";
        emit({ type: "llm_start" });
        emit({ type: "state", state: "rewriting" });
        break;
      case "llm_token":
        emit({ type: "llm_token", text: payload.text ?? "" });
        break;
      case "llm_end":
        emit({ type: "llm_end", text: payload.text ?? "" });
        emit({ type: "state", state: "done" });
        break;
      default:
        break;
    }
  };

  return {
    onEvent(cb) {
      listeners.push(cb);
    },

    async spawn() {
      const { listen } = await import("@tauri-apps/api/event");
      const events = [
        "asr_ready", "asr_partial", "asr_final",
        "llm_start", "llm_token", "llm_end",
      ];
      for (const eventName of events) {
        const unlisten = await listen(eventName, (event) => {
          handleEvent(eventName, event.payload);
        });
        unlistenFns.push(unlisten);
      }
      emit({ type: "state", state: "idle" });
    },

    kill() {
      unlistenFns.forEach((un) => un());
      unlistenFns = [];
      listeners = [];
    },

    start() {
      (async () => {
        const { invoke } = await import("@tauri-apps/api/core");
        try {
          await invoke("start_listening");
          currentText = "";
          nextUtteranceId += 1;
        } catch (e) {
          emit({ type: "state", state: "error" });
        }
      })();
    },

    stop() {
      (async () => {
        const { invoke } = await import("@tauri-apps/api/core");
        try {
          await invoke("stop_listening");
          emit({ type: "state", state: "done" });
        } catch (e) {
          emit({ type: "state", state: "error" });
        }
      })();
    },

    sendCommand(cmd: Record<string, unknown>) {
      (async () => {
        const { invoke } = await import("@tauri-apps/api/core");
        try {
          if (cmd.type === "start_recording") {
            await invoke("start_listening");
          } else if (cmd.type === "stop_recording") {
            await invoke("stop_listening");
          }
        } catch (e) {
          emit({ type: "state", state: "error" });
        }
      })();
    },
  };
}
