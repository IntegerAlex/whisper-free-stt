use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ModelManifest {
    pub id: &'static str,
    pub name: &'static str,
    pub url: &'static str,
    pub size_bytes: u64,
    pub backend: &'static str,
    pub recommended: bool,
    pub is_archive: bool,
    pub filename: Option<&'static str>,
}

pub const MODEL_MANIFEST: &[ModelManifest] = &[
    ModelManifest {
        id: "silero-vad",
        name: "Silero VAD",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/silero_vad.onnx",
        size_bytes: 1_000_000,
        backend: "vad",
        recommended: true,
        is_archive: false,
        filename: Some("silero_vad.onnx"),
    },
    ModelManifest {
        id: "parakeet-tdt-0.6b-v2-int8",
        name: "Parakeet TDT 0.6B v2 (int8)",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8-2025-01-29.tar.bz2",
        size_bytes: 1_000_000_000,
        backend: "parakeet",
        recommended: true,
        is_archive: true,
        filename: None,
    },
    ModelManifest {
        id: "whisper-large-v3-turbo-q5_1",
        name: "Whisper large-v3-turbo (Q5_1)",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-large-v3-turbo-q5_1-2024-11-20.tar.bz2",
        size_bytes: 6_000_000_000,
        backend: "whisper",
        recommended: false,
        is_archive: true,
        filename: None,
    },
    ModelManifest {
        id: "whisper-base-q5_1",
        name: "Whisper base (Q5_1)",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base-q5_1-2024-11-20.tar.bz2",
        size_bytes: 750_000_000,
        backend: "whisper",
        recommended: false,
        is_archive: true,
        filename: None,
    },
    ModelManifest {
        id: "gemma-3-1b-it-q4_k_m",
        name: "Gemma 3 1B IT (Q4_K_M)",
        url: "https://huggingface.co/unsloth/gemma-3-1b-it-gguf/resolve/main/gemma-3-1b-it-q4_k_m.gguf",
        size_bytes: 500_000_000,
        backend: "llm",
        recommended: true,
        is_archive: false,
        filename: Some("gemma-3-1b-it-q4_k_m.gguf"),
    },
];

pub fn find_model(id: &str) -> Option<&'static ModelManifest> {
    MODEL_MANIFEST.iter().find(|m| m.id == id)
}

#[allow(dead_code)]
pub fn model_path(models_dir: &Path, model: &ModelManifest) -> PathBuf {
    models_dir.join(model.id)
}

pub struct ModelManager {
    model_dir: PathBuf,
}

impl ModelManager {
    pub fn new(model_dir: PathBuf) -> Self {
        Self { model_dir }
    }

    pub fn status(&self) -> Vec<ModelStatus> {
        let mut statuses = Vec::new();

        for model in MODEL_MANIFEST {
            let model_dir = self.model_dir.join(model.id);
            let downloaded = self.verify(&model.id);
            let (downloaded_flag, size_bytes) = if model_dir.exists() {
                let total = walk_dir_size(&model_dir).unwrap_or(0);
                (true, total)
            } else {
                (false, 0)
            };
            statuses.push(ModelStatus {
                name: model.name.to_string(),
                id: model.id.to_string(),
                downloaded: downloaded || downloaded_flag,
                path: model_dir.to_string_lossy().to_string(),
                size_bytes,
                url: model.url.to_string(),
                backend: model.backend.to_string(),
                recommended: model.recommended,
            });
        }

        statuses
    }

    pub fn verify(&self, id: &str) -> bool {
        let model_opt = find_model(id);
        if let Some(model) = model_opt {
            verify_model(&self.model_dir, model)
        } else {
            false
        }
    }

    pub async fn download(&self, id: &str, mut progress: impl FnMut(usize, u64)) -> Result<()> {
        let model_opt = find_model(id);
        if let Some(model) = model_opt {
            download_model(model, &self.model_dir, |p, b| progress(p, b)).await
        } else {
            Err(anyhow::anyhow!("Model not found: {}", id))
        }
    }
}

/// Stream an HTTP response body into `dest`, reporting progress as
/// `(percent, downloaded_bytes)`. Shared by the archive and single-file
/// branches of [`download_model`].
async fn stream_to_file(
    response: reqwest::Response,
    dest: &Path,
    total: u64,
    progress: &mut impl FnMut(usize, u64),
) -> Result<()> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    let mut file = tokio::fs::File::create(dest).await?;
    let mut downloaded: u64 = 0;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| anyhow::anyhow!("Stream error: {}", e))?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        let percent = (downloaded as f64 / total as f64 * 100.0) as usize;
        progress(percent, downloaded);
    }
    file.flush().await?;
    drop(file);

    Ok(())
}

pub async fn download_model(
    model: &ModelManifest,
    target_dir: &Path,
    mut progress: impl FnMut(usize, u64),
) -> Result<()> {
    let url = model.url;
    let model_dir = target_dir;

    if model_dir.join(".downloaded").exists() {
        return Ok(());
    }

    std::fs::create_dir_all(model_dir)?;

    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to start download: {}", e))?;

    let total = response.content_length().unwrap_or(model.size_bytes);

    let (final_path, _is_archive) = if model.is_archive {
        let archive_path = model_dir.join("model.tar.bz2");
        stream_to_file(response, &archive_path, total, &mut progress).await?;

        let tar_bytes = std::fs::read(&archive_path)?;
        let decompressed = bzip2::read::BzDecoder::new(&tar_bytes[..]);
        let mut archive = tar::Archive::new(decompressed);
        archive.unpack(model_dir)?;
        std::fs::remove_file(&archive_path)?;

        (model_dir.to_path_buf(), true)
    } else {
        let ext = model.url.rsplit('.').next().unwrap_or("bin");
        let file_name: String = model.filename
            .map(|f| f.to_string())
            .unwrap_or_else(|| format!("{}.{}", model.id, ext));
        let file_path = model_dir.join(&file_name);
        stream_to_file(response, &file_path, total, &mut progress).await?;

        (file_path, false)
    };

    let _ = (final_path, _is_archive);
    std::fs::write(model_dir.join(".downloaded"), b"")?;

    Ok(())
}

pub fn verify_model(models_dir: &Path, model: &ModelManifest) -> bool {
    let model_dir = models_dir.join(model.id);
    if !model_dir.exists() {
        return false;
    }

    match model.backend {
        "vad" => {
            model_dir.join("silero_vad.onnx").exists()
        }
        "parakeet" => {
            model_dir.join("encoder.onnx").exists()
                && model_dir.join("decoder.onnx").exists()
                && model_dir.join("joiner.onnx").exists()
                && model_dir.join("tokens.txt").exists()
        }
        "whisper" => {
            model_dir.join("whisper-encoder.onnx").exists()
                && model_dir.join("whisper-decoder.onnx").exists()
                && model_dir.join("vocabulary.json").exists()
        }
        "llm" => {
            model_dir.join(format!("{}.gguf", model.id)).exists()
        }
        _ => false,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub name: String,
    pub id: String,
    pub downloaded: bool,
    pub path: String,
    pub size_bytes: u64,
    pub url: String,
    pub backend: String,
    pub recommended: bool,
}

pub(crate) fn walk_dir_size(path: &std::path::Path) -> Result<u64, anyhow::Error> {
    let mut total = 0u64;
    if path.is_dir() {
        for entry in std::fs::read_dir(path).map_err(|e| anyhow::anyhow!(e))? {
            let entry = entry.map_err(|e| anyhow::anyhow!(e))?;
            let meta = entry.metadata().map_err(|e| anyhow::anyhow!(e))?;
            if meta.is_file() {
                total += meta.len();
            } else if meta.is_dir() {
                total += walk_dir_size(&entry.path())?;
            }
        }
    }
    Ok(total)
}
