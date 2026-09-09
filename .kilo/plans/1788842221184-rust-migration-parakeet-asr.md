# Plan: Rust-Migrate + Parakeet ASR Upgrade

## Goal

Migrate Floure STT desktop app to a pure-Rust Tauri binary (eliminating the
PyInstaller Python sidecar). Primary ASR: NVIDIA Parakeet TDT via sherpa-onnx
`OnlineRecognizer` (true streaming, partial hypotheses). Fallback: Whisper via
sherpa-onnx `OfflineRecognizer` (multilingual). Local LLM cleanup: Gemma 3 1B
via `llama-cpp-4`. UI/UX: real-time partial streaming display, model management
overhaul, cross-platform polish (Windows/macOS/Linux).

## Resolved Decisions

1. **ASR backend**: sherpa-onnx Rust crate (v1.10+) — provides `OnlineRecognizer`
   (streaming Parakeet TD T), `OfflineRecognizer` (Whisper fallback), `SileroVad`,
   and `LinearResampler` all in one dependency. Eliminates need for separate
   VAD crate or audio resampling crate.
2. **Primary model**: `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2` (int8, ~1 GB) —
   streaming transducer via `OnlineRecognizer`. Real-time partial hypotheses
   appear as you speak.
3. **Fallback model**: `sherpa-onnx-whisper-large-v3-turbo` (Q5_1, ~6 GB) —
   multilingual, offline via `OfflineRecognizer`. Used only for non-English.
4. **Local LLM**: `gemma-3-1b-it-q4_k_m.gguf` (~450 MB) via `llama-cpp-4` crate.
   Default cleanup model. DeepSeek/OpenRouter retained as premium HTTP option.
5. **Audio I/O**: `cpal` crate — default host per platform (ALSA/WASAPI/CoreAudio).
   Use sherpa-onnx `LinearResampler` for 48kHz→16kHz instead of a separate crate.
6. **Binary architecture**: Zero Python. All ASR, VAD, LLM, audio, output in Rust.
   Removes PyInstaller sidecar, `stt/` package, `pyproject.toml`, `uv.lock`.

## Data Flow

```
cpal input callback (48kHz f32 stereo)
    → channel mpsc::Sender<Vec<f32>>
    → audio thread
        → LinearResampler (48k→16k mono f32)
        → push to ring buffer (Vec<f32>)
            → VAD: SileroVad, accept_waveform in 512-sample windows
                → speech segment buffer (when speech detected)
                    → OnlineRecognizer (Parakeet): create_stream + accept_waveform + decode
                        → partial hypotheses every N ms → Tauri event "llm_partial"
                        → on VAD endpoint: final result → Tauri event "llm_final"
                            → LLM cleanup (llama-cpp-4, streaming) → Tauri event "llm_token"
                                → typing (wtype/xdotool/ydotool/osascript) + clipboard
                                → SQLite history save
```

### Threading model

- **Audio callback thread** (cpal): reads mic samples, channels to `mpsc::Receiver`
- **Pipeline thread** (tokio `spawn_blocking` or std `thread`): consumes audio,
  runs VAD + ASR, emits Tauri events via `app_handle.emit_all("asr_partial", ...)`
- **LLM thread** (tokio task): consumes final transcription, streams cleanup tokens
- **Output**: synchronous typing/clipboard calls within LLM emit callback

This matches the sherpa-onnx example pattern (single mpsc channel + loop thread)
but wrapped in Tauri's async runtime via `tauri::async_runtime::spawn`.

## Affected Boundaries

### Deleted
- Entire `stt/` Python package (engine, cli, config, audio_capture, vad, transcription, llm, prompts, orchestrator, server, history, embeddings, typing, clipboard, _cpp_worker, _whisper_worker)
- `stt-engine.spec`
- `stt-ui/scripts/build-sidecar.sh`
- `pyproject.toml`, `uv.lock`
- `stt.egg-info/`, `stt/__pycache__/`, all `__pycache__` dirs
- `build/stt-engine/` (PyInstaller build artifacts)

### Replaced
- Python audio capture → `asr/audio.rs` (cpal + sherpa-onnx LinearResampler)
- Python VAD → `asr/vad.rs` (sherpa-onnx SileroVad, port of StreamingEndpointDetector)
- Python ASR (transcription.py + _cpp_worker.py) → `asr/parakeet.rs` + `asr/whisper.rs`
- Python LLM (llm.py + prompts.py) → `asr/llm.rs` (llama-cpp-4 + HTTP fallback)
- Python history (history.py + insights.py + embeddings.py) → extend `lib.rs` SQLite layer
- Python typing/clipboard → `asr/output.rs` (extend existing Win32 in lib.rs to non-Windows)
- Python orchestrator (orchestrator.py + server.py) → `asr/pipeline.rs`

### Modified
- `stt-ui/src-tauri/Cargo.toml` — add crates, remove `tauri-plugin-sql`? (NO: keep for existing history queries, it's already working)
- `stt-ui/src-tauri/tauri.conf.json` — remove `externalBin: ["binaries/stt-engine"]`, update bundle targets
- `.github/workflows/release.yml` — remove Python setup + sidecar build; remove appimagetool
- `.github/workflows/ci.yml` — remove Python lint/test jobs; add Rust `cargo fmt --check` + `cargo clippy`
- `stt-ui/src/store.ts` — replace MODEL_CATALOG, add LLM provider config
- `stt-ui/src/App.tsx` — add streaming event handlers for `asr_partial`, `llm_token`
- `README.md` — new architecture + model list

## Module Structure (new files under `stt-ui/src-tauri/src/`)

```
src-tauri/src/
├── lib.rs           # (exists) — extend: register pipeline commands
├── main.rs          # (exists) — no change needed
├── audio.rs         # cpal device enumeration + input stream + resample
├── vad.rs           # SileroVad wrapper + StreamingEndpointDetector port
├── parakeet.rs      # OnlineRecognizer wrapper (streaming Parakeet)
├── whisper.rs       # OfflineRecognizer wrapper (Whisper fallback)
├── llm.rs           # llama-cpp-4 local + HTTP fallback, prompt constants
├── pipeline.rs      # StreamingPipeline struct: audio→VAD→ASR→LLM→output
├── models.rs        # Model manifest, download (reqwest + progress), cache paths
├── config.rs        # Settings schema: profiles, LLM provider, output prefs
└── output.rs        # Typing (wtype/osascript/xdotool/ydottool) + clipboard (wl-copy/xclip/pbcopy)
```

### Module responsibilities

#### `audio.rs`
- `list_input_devices() -> Vec<(String, DeviceId)>`
- `start_capture(device_id: Option<&str>, tx: Sender<Vec<f32>>) -> Result<Stream>`
- Uses `cpal::default_host()`, `host.default_input_device()`, `device.default_input_config()`
- Handles `SampleFormat::F32 | I16 | U16` (all three conversion paths from example)
- Returns raw 48kHz f32; resampling happens in pipeline thread

#### `vad.rs`
- `struct SileroVadWrapper { vad: VoiceActivityDetector }`
- Port `StreamingEndpointDetector` logic from `stt/vad.py`:
  - `calibrate(&mut self, samples: &[f32])` — RMS threshold learning (1.5× multiplier)
  - `push(&mut self, samples: &[f32])` — feed to Silero + ring buffer
  - `poll_segment(&mut self) -> Option<Vec<f32>>` — returns speech segment when VAD `is_empty()` is false
- Config: threshold=0.5, min_silence=0.25s, min_speech=0.25s, max_speech=5.0s, window=512

#### `parakeet.rs`
- `struct ParakeetRecognizer { recognizer: OnlineRecognizer, stream: OnlineStream }`
- `fn new(model_path: &Path, num_threads: i32) -> Result<Self>`
- Config: `OnlineModelConfig` with `transducer.encoder/decoder/joiner`, `tokens`, `model_type="nemo_transducer"`, `num_threads`
- `OnlineRecognizerConfig` defaults for endpoint detection: `enable_endpoint=true`, `rule1_min_trailing_silence=2.4`, `rule2_min_trailing_silence=1.2`, `rule3_min_utterance_length=20.0`
- `fn accept(&mut self, samples: &[f16])` — `stream.accept_waveform(16000, samples)`, `recognizer.decode(&stream)`
- `fn get_partial(&self) -> Option<String>` — `recognizer.get_result(&stream).text` (partial hypothesis)
- `fn is_endpoint(&self) -> bool` — `recognizer.is_endpoint(&stream)`
- `fn reset(&self)` — `recognizer.reset(&stream)` for next utterance

#### `whisper.rs`
- `struct WhisperRecognizer { recognizer: OfflineRecognizer }`
- `fn transcribe(&self, samples: &[f32]) -> String`
- Config: `OfflineRecognizerConfig` with `whisper` model config (encoder, decoder, vocabulary)

#### `llm.rs`
- `enum LlmBackend { Local(LlamaModel), Cloud(HttpClient) }`
- `struct LlmCleanup { backend: LlmBackend }`
- `fn new(provider: &str, model_path: Option<&Path>) -> Result<Self>`
- Prompt constants ported from `stt/prompts.py` — exact system prompt strings, junk filter regex patterns
- `fn stream_cleanup(&mut self, text: &str, callback: impl FnMut(String))` — streaming token-by-token
- Junk filter: lines starting with `<`, `[`, "Note:", "Here" → filtered

#### `models.rs`
- `struct ModelManifest { id: String, name: String, url: String, size: u64, backend: "parakeet"|"whisper"|"gemma" }`
- `MODEL_MANIFEST: &[ModelManifest]` — hardcoded list of all supported models with HF URLs
- `fn download_model(model: &ModelManifest, progress: impl FnMut(usize, u64)) -> Result<PathBuf>` — reqwest streaming GET, write to `~/.local/share/floure/models/<id>/`
- `fn verify_model(id: &str) -> bool` — check all required files exist

#### `pipeline.rs`
- `struct StreamingPipeline { parakeet: ParakeetRecognizer, vad: SileroVadWrapper, resampler: Option<LinearResampler>, llm: LlmCleanup, buffer: Vec<f32>, offset: usize, config: AppConfig }`
- `fn new(config: AppConfig) -> Result<Self>` — loads models lazily on first use
- `fn run(&mut self, app_handle: AppHandle) -> Result<()>` — main loop:
  ```
  loop {
      rx.recv() → samples
      → resample 48k→16k
      → buffer.extend(resampled)
      → while offset + 512 <= buffer.len(): vad.accept(&buffer[offset..offset+512])
      → if vad segment complete: parakeet.decode_segment → emit "asr_partial" events live → on endpoint: emit "llm_final" → llm.stream_cleanup → emit "llm_token" events → output.typing + clipboard + history.save
  }
  ```

#### `output.rs`
- `fn type_text(text: &str, platform: &str)` — wtype (Linux), osascript (macOS), SendInput (Windows, extend existing)
- `fn copy_to_clipboard(text: &str)` — wl-copy/xclip/pbcopy/clip
- `fn save_history(transcript: &str, source: &str)` — insert into SQLite (extend existing queries in lib.rs)

## ASR Profiles (new)

| Profile | Backend | Model | Size | Use |
|---|---|---|---|---|
| `parakeet` | OnlineRecognizer | `nemo-parakeet-tdt-0.6b-v2-int8` | ~1 GB | Default, streaming, best English |
| `whisper-turbo` | OfflineRecognizer | `whisper-large-v3-turbo-q5_1` | ~6 GB | Multilingual fallback (GPU optional) |
| `whisper-base` | OfflineRecognizer | `whisper-base-q5_1` | ~750 MB | General purpose fallback |

Profile `auto`: Parakeet for English; Whisper large-v3-turbo for non-English with GPU;
Whisper base-q5_1 for non-English CPU-only.

## Tasks (ordered, implementation-ready)

### Phase 0 — Foundation (3-4 tasks)
0. **`Cargo.toml`**: Add `sherpa-onnx`, `cpal`, `llama-cpp-4`, `reqwest` (json+stream), `anyhow`, `f16`. Keep `tauri-plugin-sql`, `tauri-plugin-fs`, `tauri-plugin-dialog`, `tauri-plugin-process`. Do NOT remove existing crates — they still support Rust-side history/clipboard.
1. **Module skeleton**: Create `audio.rs`, `vad.rs`, `parakeet.rs`, `whisper.rs`, `llm.rs`, `models.rs`, `pipeline.rs`, `output.rs`, `config.rs` with stub `pub fn` signatures. Add `mod` declarations in `lib.rs`.
2. **`config.rs`**: Define `AppConfig` struct mirroring `store.ts` SettingsSchema: profile, llm provider, selected mic, output prefs, model dir path. Port from `stt/config.py` `TranscriptionConfig`.
3. **Update `AGENTS.md`**: Add Rust toolchain note (`rust-toolchain.toml` → 1.85+ for cpal MSRV), add sherpa-onnx build note.

### Phase 1 — ASR Backend — sherpa-onnx (4-5 tasks)
4. **`models.rs` model manifest**: Hardcode all model URLs (HF Hub paths + prebuilt archive names). Implement `download_model` with `reqwest::get` stream + progress callback + `tokio::fs::write`. Implement `verify_model`.
5. **`parakeet.rs`**: Implement `ParakeetRecognizer::new` using `OnlineRecognizerConfig` + `OnlineModelConfig` with `nemo_transducer` type. Match exact field names from sherpa-onnx Rust docs.
6. **`parakeet.rs`**: Implement `accept`, `get_partial`, `is_endpoint`, `reset` using `OnlineStream` + `recognizer.decode`.
7. **`whisper.rs`**: Implement `WhisperRecognizer` using `OfflineRecognizerConfig` with Whisper model config (encoder, decoder, vocabulary).
8. **Tests**: Model loading (mocked file check), transcription correctness on test WAV, language fallback for unsupported locales.

### Phase 2 — Audio + VAD (cpal + Silero) (4 tasks)
9. **`audio.rs`**: Implement device enumeration + input stream using `cpal::traits::{DeviceTrait, HostTrait, StreamTrait}`. Handle F32/I16/U16 formats (copy from sherpa-onnx example). Return `mpsc::Receiver<Vec<f32>>` or `Arc<Mutex<Vec<f32>>>`.
10. **`vad.rs`**: Implement `SileroVadWrapper` with `VadModelConfig` defaults (threshold=0.5, min_silence=0.25, min_speech=0.25, max_speech=5.0, window=512, sample_rate=16000). Port `StreamingEndpointDetector` from `stt/vad.py`.
11. **Tests**: Ring buffer overflow behavior, VAD detection of speech/non-speech, endpointing edge cases (short utterances, long pauses).

### Phase 3 — LLM Cleanup (llama-cpp-4) (3 tasks)
12. **`llm.rs`**: Implement `LlmCleanup::new` — load Gemma 3 1B Q4_K_M via `llama_cpp_4::LlamaModel::load`. Use `DefaultParams` + `n_ctx=512` + `n_threads=4`.
13. **`llm.rs`**: Implement `stream_cleanup` — create prompt string (from `stt/prompts.py` constants), stream `model.decode()` tokens via callback. Port `_clean_response` junk filtering.
14. **`llm.rs`**: Implement `LlmBackend::Cloud` for DeepSeek/OpenRouter HTTP with `reqwest` + SSE parsing.
15. **Tests**: Prompt construction matches Python original, streaming token yields, junk filtering removes expected line types.

### Phase 4 — Pipeline Orchestration (4 tasks)
16. **`pipeline.rs`**: Implement `StreamingPipeline::new` — initialize Parakeet (lazy), SileroVAD, LLM, resampler. Match `streaming_simulate` function structure from sherpa-onnx example.
17. **`pipeline.rs`**: Implement `run` loop — cpal callback → recv channel → resample → buffer → VAD window feeding (512 samples) → interim decode on speech → final decode on endpoint → LLM stream → output → history.
18. **`output.rs`**: Implement `type_text` cross-platform (extend existing Win32 `win32_send_keystrokes` in lib.rs to `wtype`/`ydotool`/`osascript`). Implement `copy_to_clipboard` (`wl-copy`/`xclip`/`pbcopy`/`clip`).
19. **`pipeline.rs`**: Wire Tauri event emission — `app.emit_all("asr_partial", {...})`, `app.emit_all("llm_final", {...})`, `app.emit_all("llm_token", {...})`, `app.emit_all("asr_error", {...})`.
20. **Tests**: End-to-end with mock audio buffer (silent WAV → no output, speech WAV → correct text), latency tracking (<1s ASR, <3s LLM).

### Phase 4.5 — Tauri Integration (3 tasks)
21. **`lib.rs`**: Register `start_listening`, `stop_listening`, `test_microphone`, `get_available_mics`, `download_model` commands. Wire to `setup()` or tray menu.
22. **`tauri.conf.json`**: Remove `externalBin: ["binaries/stt-engine"]` entry. Ensure `beforeBuildCommand` is `"pnpm run build"` only (no sidecar build). Add Linux `apt` dependencies: `libasound2-dev`, `libdbus-1-dev`.
23. **CI**: Update `.github/workflows/ci.yml` and `release.yml` — remove Python jobs, add Rust toolchain + `cargo clippy` + `cargo fmt --check`.

### Phase 5 — UI Refresh (6 tasks)
24. **`store.ts`**: Replace `MODEL_CATALOG` with new profiles (parakeet, whisper-turbo, whisper-base, gemma-3-1b). Add `LlmProvider` type (`"local" | "deepseek" | "openrouter"`). Add `Profile` type union.
25. **`App.tsx`**: Wire `.listen("asr_partial", ...)` → update live transcript display. Wire `llm_token` → token-by-token reveal.
26. **Componentize**: Split `App.tsx` into `TranscriptionFeed`, `LiveMeter`, `LatencyOverlay`, `ModelManager`, `OnboardingFlow`.
27. **`ModelsPage`**: List new models with download progress (reuse `modelDownloadProgress` reducer pattern from `OnboardingState`).
28. **Windows fix**: Complete Win32 focus + typing gaps per issue #13.
29. **`store.ts` AppStateReducer**: Add `SET_LLM_PROVIDER`, `SET_PROFILE` actions.

### Phase 6 — Cleanup & Packaging (3 tasks)
30. Delete all Python files, specs, locks, `__pycache__`, `stt/` package, `build/` artifacts.
31. Update `README.md` — new architecture diagram, model list, local LLM instructions.
32. Update CI workflows — remove PyInstaller, appimagetool, Python setup.

### Phase 7 — Validation (3 tasks)
33. **Lint**: `cargo clippy -- -D warnings`, `cargo fmt --check`, `pnpm run lint`, `pnpm run test`.
34. **Benchmark**: ASR latency <1s (Parakeet CPU), LLM cleanup <3s (Gemma 3 streaming), binary size <150 MB.
35. **WER smoke**: `tests/test_audio_speech.wav` — Parakeet vs old Whisper baseline. Cross-platform dev builds on Linux/Wayland, Windows, macOS.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| sherpa-onnx Rust crate static linking fails in CI | Pin crate version + `SHERPA_ONNX_LIB_DIR` env. Use `build.rs` auto-download feature as fallback. Cache in GH Actions `~/.cache/sherpa-onnx/`. |
| cpal requires `libasound2-dev` on Linux CI | Add `apt-get install -y libasound2-dev libdbus-1-dev` to CI workflow for Linux builds. |
| llama-cpp-4 ABI mismatch with llama.cpp | Pin `llama-cpp-4` to a specific version matching a known-good llama.cpp commit. Use `CMAKE` feature gate. |
| Parakeet model download is 1 GB | Lazy download on first run. Show progress bar in `OnboardingFlow`. Allow user to place models manually. |
| Existing user history DB at `~/.local/share/stt/history.db` → new path `~/.local/share/floure/` | Schema is identical (same columns). On startup, check old path; if exists and new doesn't, copy file. Log migration. |
| Binary size with ONNX runtime + llama.cpp statically linked | `cargo build --release` with `strip`. Target: separate `llama.cpp` from ONNX if size exceeds 200 MB. |
| Real-time partial streaming causes frontend lag | Batch `asr_partial` events (max 2/sec), use `Arc<String>` to avoid clones. Emit `llm_token` only for non-whitespace deltas. |
| Windows typing focus (issue #13) | Existing `win32` module in lib.rs handles Win32 `SendInput` — just needs `get_foreground_hwnd` fix. Already partially done. |

## Out of Scope (v1.1 candidates)

- Embedding-based few-shot context (`stt/embeddings.py`) — defer; use dictionary + recency context
- WebSocket server (`stt/server.py`) — Floure is desktop-only; no server needed
- AppImage/Linux packaging format changes — keep current Tauri bundler (deb/rpm/dmg/exe)
- Cloud ASR (OpenAI Whisper API, Azure STT) — local-first; premium LLM HTTP is the only cloud component

## Validation Plan

- **Unit tests in Rust** (`cargo test`): ASR model loading, VAD correctness, LLM prompt construction, dictionary replacement, ring buffer
- **Frontend tests** (`pnpm run test`): Streaming event handlers, model download progress UI, profile selector
- **Benchmark**: `cargo run --release -- example_parakeet` with 10s test audio — measure: (1) time from utterance end to final text (<2s target), (2) time to LLM cleanup completion (<5s target), (3) binary size (<150 MB target)
- **WER smoke**: Parakeet vs old Whisper baseline on `tests/test_audio_speech.wav` — target ≥5% WER improvement on English dictation
- **Cross-platform**: Linux (Wayland+X11, PipeWire+ALSA), Windows (WASAPI), macOS (CoreAudio) dev builds
- **CI gate**: `cargo fmt --check` + `cargo clippy -- -D warnings` + `pnpm run lint` + `pnpm run test` must all pass

## Open Questions

1. **true streaming vs simulated**: `OnlineRecognizer` with streaming Parakeet requires streaming-specific ONNX weights. If streaming weights aren't available in sherpa-onnx, fall back to simulated streaming (VAD + OfflineRecognizer with interim decode — matches sherpa-onnx example exactly). Decision: try streaming first; if model not available, use simulated per example.
2. **llama-cpp-4 vs llm-chain**: `llama-cpp-4` provides raw model inference; `llm-chain` or `burn` would add abstraction. Decision: use `llama-cpp-4` directly — simplest, lowest dependency bloat, full control over token streaming.
3. **Model download location**: `~/.local/share/floure/models/` (Linux), `%APPDATA%\floure\models` (Windows), `~/Library/Application Support/floure/models` (macOS). Use `tauri::api::path::app_data_dir()` for cross-platform.
