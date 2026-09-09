# Floure — Project Context

## What it is

Floure (repo `stt`) is a Linux-first desktop dictation app: speak into the
mic → speech is transcribed locally → an LLM pass cleans it up → text is
typed into the focused window or copied to the clipboard. No cloud account
required for the local path.

## Stack

- **Frontend:** Tauri v2 + React 19 (`stt-ui/src`), TypeScript, Vite.
- **Backend:** Rust inside `stt-ui/src-tauri/src` (Tauri commands in `lib.rs`).
  - Audio capture: `cpal` (`audio.rs`).
  - VAD: Silero (`vad.rs`).
  - ASR: Parakeet + Whisper via `sherpa-onnx` (`parakeet.rs`, `whisper.rs`, `models.rs`).
  - LLM cleanup: local Gemma 3 via `llama.cpp`, or DeepSeek / OpenRouter cloud (`llm.rs`).
  - Output: typing + clipboard (`output.rs`); end-to-end loop in `pipeline.rs`.

## Key paths

- UI: `stt-ui/src` · Rust backend: `stt-ui/src-tauri/src` · Tauri config: `stt-ui/src-tauri/tauri.conf.json`
- History DB: `~/.local/share/floure/history.db` (`STT_DATA_DIR` overrides it;
  legacy `~/.local/share/stt/history.db` is migrated forward) — see `docs/adr/0002-history-db-path.md`.
- Config: `~/.config/floure/config.json`.
- Decisions: `docs/adr/` (Rust-native backend, history DB path).

## Build / check / test

```bash
cd stt-ui && pnpm install && pnpm dev     # frontend dev server
cd stt-ui && pnpm build                   # frontend build
cd stt-ui/src-tauri && cargo check        # Rust check
cd stt-ui/src-tauri && cargo test         # Rust tests
cd stt-ui && pnpm tauri build             # full desktop build
```

## Glossary

- **Profile:** ASR model preset (Parakeet / Whisper Turbo / Whisper Base).
- **Cleanup:** LLM rewrite pass over raw transcripts (punctuation, fillers).
- **Pipeline:** mic → VAD → ASR → cleanup → output loop (`pipeline.rs`).
