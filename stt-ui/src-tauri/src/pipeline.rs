use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;
use tauri::Emitter;

use crate::config::{AppConfig, history_db_path};
use crate::llm::{LlmBackend, LlmCleanup, LlmMode};
use crate::models::{MODEL_MANIFEST, find_model, download_model, verify_model};
use crate::parakeet::ParakeetRecognizer;
use crate::vad::VoiceActivityDetector;
use crate::whisper::WhisperRecognizer;
use crate::output::{save_to_history, type_text, copy_to_clipboard};

static PIPELINE_RUNNING: std::sync::OnceLock<Arc<AtomicBool>> = std::sync::OnceLock::new();

pub fn get_running_flag() -> &'static Arc<AtomicBool> {
    PIPELINE_RUNNING.get_or_init(|| Arc::new(AtomicBool::new(false)))
}

pub struct LlmProcessor {
    llm: Option<LlmCleanup>,
    config: AppConfig,
}

impl LlmProcessor {
    pub fn new(config: AppConfig) -> Self {
        let llm_model_path = config.model_dir.join("gemma-3-1b-it-q4_k_m.gguf");
        let llm = match LlmCleanup::new(LlmBackend::Local, Some(&llm_model_path)) {
            Ok(l) => Some(l),
            Err(e) => {
                None
            }
        };
        Self { llm, config }
    }

    pub fn process(&mut self, text: &str, app: &tauri::AppHandle) {
        let mode = self.config.llm_mode;
        let cleaned: String;

        if mode == LlmMode::Off {
            cleaned = text.to_string();
        } else if let Some(llm) = &mut self.llm {
            let prompt = crate::llm::build_prompt(text, mode, "", "");
            let _ = app.emit("llm_start", serde_json::json!({}));

            let collected = Arc::new(std::sync::Mutex::new(String::new()));
            let collected_clone = collected.clone();
            let app_for_callback = app.clone();

            let result = llm.stream_cleanup(&prompt, move |token| {
                let _ = app_for_callback
                    .emit("llm_token", serde_json::json!({ "token": token }));
                collected_clone.lock().unwrap().push_str(&token);
            });

            let _ = app.emit("llm_end", serde_json::json!({}));
            if result.is_ok() {
                cleaned = crate::llm::clean_response(&collected.lock().unwrap());
            } else {
                cleaned = crate::llm::clean_response(text);
            }
        } else {
            cleaned = crate::llm::clean_response(text);
        }

        if self.config.typing_enabled {
            let _ = type_text(&cleaned);
        }
        if self.config.clipboard_enabled {
            let _ = copy_to_clipboard(&cleaned);
        }

        let db_path = history_db_path();
        let _ = save_to_history(&cleaned, text, mode.as_str(), "floure", &db_path);
    }
}

pub struct PipelineController {
    running: Arc<AtomicBool>,
    app: tauri::AppHandle,
    config: AppConfig,
    llm_processor: LlmProcessor,
    silero_path: PathBuf,
    model_dir: PathBuf,
}

impl PipelineController {
    pub fn new(app: tauri::AppHandle, config: AppConfig) -> Result<Self> {
        let running = get_running_flag().clone();
        let model_dir = config.model_dir.clone();
        let silero_path = model_dir.join("silero-vad").join("silero_vad.onnx");

        if !silero_path.exists() {
            std::fs::create_dir_all(&model_dir.join("silero-vad"))?;
            let model_dir_dl = model_dir.join("silero-vad");
            let vad_model = MODEL_MANIFEST
                .iter()
                .find(|m| m.id == "silero-vad")
                .unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let result = runtime.block_on(async {
                download_model(vad_model, &model_dir_dl, |_, _| {}).await
            });
            if let Err(e) = result {
                let _ = app.emit(
                    "asr_error",
                    serde_json::json!({"error": format!("Failed to download VAD: {}", e)}),
                );
                running.store(false, Ordering::SeqCst);
                return Err(anyhow::anyhow!("VAD model download failed"));
            }
        }

        if !silero_path.exists() {
            let _ = app.emit(
                "asr_error",
                serde_json::json!({"error": "Silero VAD model file not found after download attempt"}),
            );
            running.store(false, Ordering::SeqCst);
            return Err(anyhow::anyhow!("Silero VAD model not found"));
        }

        let llm_processor = LlmProcessor::new(config.clone());

        Ok(Self {
            running,
            app,
            config,
            llm_processor,
            silero_path,
            model_dir,
        })
    }

    pub fn start(&self) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(anyhow::anyhow!("Pipeline already running"));
        }

        let (tx, rx) = mpsc::channel::<Vec<f32>>();

        let mic_name: Option<String> = self.config.selected_mic_index.and_then(|i| {
            crate::audio::list_input_devices()
                .ok()
                .and_then(|devices| devices.get(i).map(|(name, _)| name.clone()))
        });

        let app_clone = self.app.clone();
        let config_clone = self.config.clone();
        let running_clone = self.running.clone();
        let silero_path = self.silero_path.clone();

        let audio = match crate::audio::start_capture(mic_name.as_deref(), move |samples: &[f32]| {
            let _ = tx.send(samples.to_vec());
        }) {
            Ok(a) => a,
            Err(e) => {
                let _ = app_clone.emit(
                    "asr_error",
                    serde_json::json!({"error": format!("Audio device error: {}", e)}),
                );
                running_clone.store(false, Ordering::SeqCst);
                return Ok(());
            }
        };

        let mic_sample_rate = audio.sample_rate;

        std::thread::spawn(move || {
            let _audio = audio;

            let resampler: Option<sherpa_onnx::LinearResampler> = if mic_sample_rate != 16000 {
                sherpa_onnx::LinearResampler::create(mic_sample_rate as i32, 16000)
            } else {
                None
            };

            let mut vad = match VoiceActivityDetector::new(&silero_path, 0.5) {
                Ok(v) => v,
                Err(e) => {
                    let _ = app_clone.emit(
                        "asr_error",
                        serde_json::json!({"error": e.to_string()}),
                    );
                    return;
                }
            };

            let model_id = config_clone.asr_profile.model_id();
            let model_dir = config_clone.asr_profile.model_dir(&config_clone.model_dir);

            let mut parakeet: Option<ParakeetRecognizer> = None;
            let mut whisper: Option<WhisperRecognizer> = None;

            if verify_model(&config_clone.model_dir, find_model(model_id).unwrap()) {
                match config_clone.asr_profile {
                    crate::config::AsrProfile::Parakeet => {
                        match ParakeetRecognizer::new(&model_dir, 4, false) {
                            Ok(r) => {
                                parakeet = Some(r);
                                let _ = app_clone
                                    .emit("asr_ready", serde_json::json!({ "backend": "parakeet" }));
                            }
                            Err(e) => {
                                let _ = app_clone
                                    .emit("asr_error", serde_json::json!({"error": e.to_string()}));
                            }
                        }
                    }
                    crate::config::AsrProfile::WhisperTurbo | crate::config::AsrProfile::WhisperBase => {
                        match WhisperRecognizer::new(&model_dir, 4, false) {
                            Ok(r) => {
                                whisper = Some(r);
                                let _ = app_clone
                                    .emit("asr_ready", serde_json::json!({ "backend": "whisper" }));
                            }
                            Err(e) => {
                                let _ = app_clone
                                    .emit("asr_error", serde_json::json!({"error": e.to_string()}));
                            }
                        }
                    }
                }
            } else {
                let _ = app_clone.emit(
                    "asr_error",
                    serde_json::json!({"error": format!("Model {} not downloaded", model_id)}),
                );
            }

            while running_clone.load(Ordering::SeqCst) {
                match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                    Ok(samples) => {
                        let resampled: Vec<f32> = if let Some(ref r) = resampler {
                            r.resample(&samples, false)
                        } else {
                            samples
                        };

                        vad.feed(&resampled);

                        if vad.is_speech_detected() {
                            if let Some(ref mut rec) = parakeet {
                                rec.accept(&resampled);

                                if let Some(partial) = rec.get_partial() {
                                    let _ = app_clone.emit(
                                        "asr_partial",
                                        serde_json::json!({ "text": partial }),
                                    );
                                }

                                if rec.is_endpoint() {
                                    if let Some(final_text) = rec.get_partial() {
                                        let _ = app_clone.emit(
                                            "asr_final",
                                            serde_json::json!({ "text": final_text }),
                                        );
                                        rec.reset();
                                    }
                                    vad.reset_after_segment();
                                }
                            }
                        }

                        if let Some(segment) = vad.try_get_segment() {
                            if let Some(ref ws) = whisper {
                                let start = Instant::now();
                                let text = ws.transcribe(&segment);
                                let latency_ms = start.elapsed().as_millis() as u64;

                                if !text.is_empty() {
                                    let _ = app_clone.emit(
                                        "asr_final",
                                        serde_json::json!({
                                            "text": text,
                                            "latency_ms": latency_ms,
                                        }),
                                    );
                                }
                            }
                            vad.reset_after_segment();
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        });

        Ok(())
    }

    pub fn stop(&self) {
        if let Some(flag) = PIPELINE_RUNNING.get() {
            flag.store(false, Ordering::SeqCst);
        }
    }
}

pub fn start_pipeline(app: tauri::AppHandle, config: AppConfig) -> Result<()> {
    let controller = PipelineController::new(app.clone(), config.clone())?;
    controller.start()?;
    Ok(())
}

pub fn stop_pipeline() {
    if let Some(flag) = PIPELINE_RUNNING.get() {
        flag.store(false, Ordering::SeqCst);
    }
}