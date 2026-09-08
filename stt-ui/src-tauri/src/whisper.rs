use anyhow::Result;
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};

pub struct WhisperRecognizer {
    recognizer: OfflineRecognizer,
}

impl WhisperRecognizer {
    pub fn new(model_dir: &std::path::Path, num_threads: i32, debug: bool) -> Result<Self> {
        let encoder = model_dir.join("whisper-encoder.onnx");
        let decoder = model_dir.join("whisper-decoder.onnx");
        let tokens = model_dir.join("vocabulary.json");

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.whisper.encoder = Some(encoder.to_str().unwrap().into());
        config.model_config.whisper.decoder = Some(decoder.to_str().unwrap().into());
        config.model_config.whisper.language = Some("en".to_string());
        config.model_config.whisper.task = Some("transcribe".to_string());
        config.model_config.tokens = Some(tokens.to_str().unwrap().into());
        config.model_config.num_threads = num_threads;
        config.model_config.debug = debug;
        config.model_config.model_type = Some("whisper".to_string());

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow::anyhow!("Failed to create Whisper OfflineRecognizer"))?;

        Ok(Self { recognizer })
    }

    pub fn transcribe(&self, samples: &[f32]) -> String {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(16000, samples);
        self.recognizer.decode(&stream);
        stream
            .get_result()
            .map(|r| r.text)
            .unwrap_or_default()
    }
}