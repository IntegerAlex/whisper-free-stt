use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::llm::LlmMode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AsrProfile {
    Parakeet,
    WhisperTurbo,
    WhisperBase,
}

impl AsrProfile {
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Parakeet => "parakeet-tdt-0.6b-v2-int8",
            Self::WhisperTurbo => "whisper-large-v3-turbo-q5_1",
            Self::WhisperBase => "whisper-base-q5_1",
        }
    }

    pub fn model_dir(&self, base: &std::path::Path) -> std::path::PathBuf {
        base.join(self.model_id())
    }
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
    #[serde(default)]
    pub llm_mode: LlmMode,
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
            llm_mode: LlmMode::default(),
            selected_mic_index: None,
            typing_enabled: true,
            clipboard_enabled: true,
            dictation_mode: false,
            hotkey: Some("ctrl+shift+s".to_string()),
            model_dir,
        }
    }
}

/// Config directory for `floure/config.json`
/// (`~/.config/floure`, `%APPDATA%/floure`, ...).
fn config_dir() -> PathBuf {
    dirs_next::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("floure")
}

/// Canonical data directory for the app.
///
/// Honors the `STT_DATA_DIR` env override (used by the Python backend);
/// otherwise `~/.local/share/floure`.
pub fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("STT_DATA_DIR") {
        PathBuf::from(dir)
    } else {
        dirs_next::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".local/share/floure")
    }
}

/// Single source of truth for the transcript history DB path.
///
/// Migrates the legacy `~/.local/share/stt/history.db` forward: if the
/// canonical DB does not exist yet but the legacy one does (and no
/// `STT_DATA_DIR` override is set), it is copied into place.
pub fn history_db_path() -> PathBuf {
    let canonical = data_dir().join("history.db");

    if std::env::var("STT_DATA_DIR").is_err() && !canonical.exists() {
        if let Some(home) = dirs_next::home_dir() {
            let legacy = home.join(".local/share/stt/history.db");
            if legacy != canonical && legacy.exists() {
                if let Some(parent) = canonical.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if std::fs::copy(&legacy, &canonical).is_ok() {
                    eprintln!(
                        "migrated history DB from legacy path {} to {}",
                        legacy.display(),
                        canonical.display()
                    );
                }
            }
        }
    }

    canonical
}

impl AppConfig {
    pub fn load() -> Self {
        let store_path = config_dir().join("config.json");

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
        let store_path = config_dir();

        std::fs::create_dir_all(&store_path)?;
        let config_path = store_path.join("config.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&config_path, json)?;
        Ok(())
    }
}
