use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AsrProfile {
    Parakeet,
    WhisperTurbo,
    WhisperBase,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmProvider {
    Local,
    DeepSeek,
    OpenRouter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub asr_profile: AsrProfile,
    pub llm_provider: LlmProvider,
    pub selected_mic_index: Option<usize>,
    pub typing_enabled: bool,
    pub clipboard_enabled: bool,
    pub dictation_mode: bool,
    pub hotkey: Option<String>,
    pub model_dir: PathBuf,
}

impl Default for AppConfig {
    fn default() -> Self {
        let model_dir = dirs_next::data_dir()
            .unwrap_or_else(|| PathBuf::from(".local/share/floure"))
            .join("floure")
            .join("models");

        Self {
            asr_profile: AsrProfile::Parakeet,
            llm_provider: LlmProvider::Local,
            selected_mic_index: None,
            typing_enabled: true,
            clipboard_enabled: true,
            dictation_mode: false,
            hotkey: Some("ctrl+shift+s".to_string()),
            model_dir,
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        let store_path = dirs_next::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("floure")
            .join("config.json");

        if store_path.exists() {
            if let Ok(contents) = std::fs::read_to_string(&store_path) {
                if let Ok(config) = serde_json::from_str::<AppConfig>(&contents) {
                    return config;
                }
            }
        }
        Self::default()
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let store_path = dirs_next::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("floure");

        std::fs::create_dir_all(&store_path)?;
        let config_path = store_path.join("config.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&config_path, json)?;
        Ok(())
    }
}
