use anyhow::Result;
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

pub async fn download_model(
    model: &ModelManifest,
    target_dir: &Path,
    mut progress: impl FnMut(usize, u64),
) -> Result<()> {
    use tokio::io::AsyncWriteExt;

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

    let (final_path, is_archive) = if model.is_archive {
        let archive_path = model_dir.join("model.tar.bz2");
        let mut file = tokio::fs::File::create(&archive_path).await?;
        let mut downloaded: u64 = 0;

        use futures_util::StreamExt;
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
        let mut file = tokio::fs::File::create(&file_path).await?;
        let mut downloaded: u64 = 0;

        use futures_util::StreamExt;
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

        (file_path, false)
    };

    let _ = (final_path, is_archive);
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
