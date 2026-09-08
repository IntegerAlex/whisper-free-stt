use anyhow::Result;
use sherpa_onnx::{VadModelConfig, VoiceActivityDetector as SherpaVad};

pub struct VoiceActivityDetector {
    vad: SherpaVad,
    buffer: Vec<f32>,
    offset: usize,
    window_size: usize,
    speech_started: bool,
    last_partial_time: std::time::Instant,
}

impl VoiceActivityDetector {
    pub fn new(model_path: &std::path::Path, threshold: f32) -> Result<Self> {
        let mut config = VadModelConfig::default();
        config.silero_vad.model = Some(model_path.to_str().unwrap().into());
        config.silero_vad.threshold = threshold;
        config.silero_vad.min_silence_duration = 0.25;
        config.silero_vad.min_speech_duration = 0.25;
        config.silero_vad.max_speech_duration = 5.0;
        config.silero_vad.window_size = 512;
        config.sample_rate = 16000;
        config.debug = false;

        let vad = SherpaVad::create(&config, 60.0)
            .ok_or_else(|| anyhow::anyhow!("Failed to create Silero VAD"))?;

        Ok(Self {
            vad,
            buffer: Vec::new(),
            offset: 0,
            window_size: 512,
            speech_started: false,
            last_partial_time: std::time::Instant::now(),
        })
    }

    pub fn feed(&mut self, samples: &[f32]) {
        self.buffer.extend_from_slice(samples);

        while self.offset + self.window_size <= self.buffer.len() {
            self.vad
                .accept_waveform(&self.buffer[self.offset..self.offset + self.window_size]);

            if !self.speech_started && self.vad.detected() {
                self.speech_started = true;
                self.last_partial_time = std::time::Instant::now();
            }

            self.offset += self.window_size;
        }
    }

    pub fn is_speech_detected(&self) -> bool {
        self.vad.detected()
    }

    #[allow(dead_code)]
    pub fn should_interim_decode(&mut self) -> bool {
        self.speech_started
            && self.last_partial_time.elapsed().as_secs_f32() > 0.2
    }

    #[allow(dead_code)]
    pub fn reset_interim_timer(&mut self) {
        self.last_partial_time = std::time::Instant::now();
    }

    pub fn try_get_segment(&mut self) -> Option<Vec<f32>> {
        if !self.vad.is_empty() {
            if let Some(segment) = self.vad.front() {
                self.vad.pop();
                return Some(segment.samples().to_vec());
            }
        }
        None
    }

    #[allow(dead_code)]
    pub fn pending_window_start(&self) -> usize {
        self.offset
    }

    #[allow(dead_code)]
    pub fn trim_buffer(&mut self) {
        if self.offset > 0 {
            self.buffer = self.buffer[self.offset..].to_vec();
            self.offset = 0;
        }
    }

    pub fn reset_after_segment(&mut self) {
        self.buffer.clear();
        self.offset = 0;
        self.speech_started = false;
    }
}
