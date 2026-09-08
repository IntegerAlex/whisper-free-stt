use anyhow::Result;
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, OnlineStream};

pub struct ParakeetRecognizer {
    recognizer: OnlineRecognizer,
    stream: OnlineStream,
}

impl ParakeetRecognizer {
    pub fn new(model_dir: &std::path::Path, num_threads: i32, debug: bool) -> Result<Self> {
        let encoder = model_dir.join("encoder.onnx");
        let decoder = model_dir.join("decoder.onnx");
        let joiner = model_dir.join("joiner.onnx");
        let tokens = model_dir.join("tokens.txt");

        let mut config = OnlineRecognizerConfig::default();
        config.model_config.transducer.encoder = Some(encoder.to_str().unwrap().into());
        config.model_config.transducer.decoder = Some(decoder.to_str().unwrap().into());
        config.model_config.transducer.joiner = Some(joiner.to_str().unwrap().into());
        config.model_config.tokens = Some(tokens.to_str().unwrap().into());
        config.model_config.model_type = Some("nemo_transducer".to_string());
        config.model_config.num_threads = num_threads;
        config.model_config.debug = debug;

        config.enable_endpoint = true;
        config.rule1_min_trailing_silence = 2.4;
        config.rule2_min_trailing_silence = 1.2;
        config.rule3_min_utterance_length = 20.0;
        config.decoding_method = Some("greedy_search".to_string());

        let recognizer = OnlineRecognizer::create(&config)
            .ok_or_else(|| anyhow::anyhow!("Failed to create Parakeet OnlineRecognizer"))?;
        let stream = recognizer.create_stream();

        Ok(Self { recognizer, stream })
    }

    pub fn accept(&mut self, samples: &[f32]) {
        self.stream.accept_waveform(16000, samples);
        while self.recognizer.is_ready(&self.stream) {
            self.recognizer.decode(&self.stream);
        }
    }

    pub fn get_partial(&self) -> Option<String> {
        self.recognizer.get_result(&self.stream).map(|r| r.text)
    }

    pub fn is_endpoint(&self) -> bool {
        self.recognizer.is_endpoint(&self.stream)
    }

    pub fn reset(&self) {
        self.recognizer.reset(&self.stream);
    }
}
