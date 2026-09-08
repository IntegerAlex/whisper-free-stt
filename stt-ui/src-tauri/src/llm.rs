use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmMode {
    Off,
    Cleanup,
    BulletList,
    Email,
    CommitMessage,
}

impl Default for LlmMode {
    fn default() -> Self {
        // Matches the pipeline's historical behavior (always Cleanup).
        Self::Cleanup
    }
}

impl LlmMode {
    /// Storage/display label used in the history DB `mode` column.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Cleanup => "cleanup",
            Self::BulletList => "bullet_list",
            Self::Email => "email",
            Self::CommitMessage => "commit_message",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LlmBackend {
    Local,
    DeepSeek,
    OpenRouter,
}

pub const CLEANUP_PROMPT: &str = "Fix punctuation, capitalization, and remove filler words (um, uh). \
Preserve technical terms. \
IMPORTANT: Return ONLY the corrected transcript text. \
Do NOT add any labels, headers, commentary, safety ratings, or explanations.";

pub const BULLET_PROMPT: &str = "Convert the following microphone transcript into a clean, \
well-structured bulleted list. Group related points together. \
Return only the bullet list with no preamble.";

pub const EMAIL_PROMPT: &str = "Rewrite the following microphone transcript as a professional email. \
Add a short subject line in brackets at the top. \
Return only the email body with no extra preamble.";

pub const COMMIT_PROMPT: &str = "Convert the following microphone transcript into a git commit message. \
Use conventional commit format (type: short description). \
Keep the subject line under 72 characters. \
Return only the commit message with no preamble.";

fn mode_instruction(mode: LlmMode) -> &'static str {
    match mode {
        LlmMode::Off => "",
        LlmMode::Cleanup => CLEANUP_PROMPT,
        LlmMode::BulletList => BULLET_PROMPT,
        LlmMode::Email => EMAIL_PROMPT,
        LlmMode::CommitMessage => COMMIT_PROMPT,
    }
}

pub fn build_prompt(
    transcript: &str,
    mode: LlmMode,
    few_shot_context: &str,
    dictionary_context: &str,
) -> String {
    let instruction = mode_instruction(mode);
    if instruction.is_empty() {
        return transcript.to_string();
    }

    let mut parts: Vec<String> = Vec::new();
    if !few_shot_context.is_empty() {
        parts.push(few_shot_context.to_string());
    }
    if !dictionary_context.is_empty() {
        parts.push(dictionary_context.to_string());
    }
    parts.push(instruction.to_string());
    parts.push(format!("Transcript:\n{}", transcript));
    parts.join("\n\n")
}

pub fn clean_response(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty()
                && !trimmed.starts_with('<')
                && !trimmed.starts_with('[')
                && !trimmed.starts_with("Note:")
                && !trimmed.starts_with("Here")
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

use llama_cpp_4::prelude::*;

pub struct LlmCleanup {
    pub backend: LlmBackend,
    backend_handle: Option<LlamaBackend>,
    local_model: Option<LlamaModel>,
    api_key: Option<String>,
}

impl LlmCleanup {
    pub fn new(backend: LlmBackend, model_path: Option<&Path>) -> Result<Self> {
        let backend_handle = if backend == LlmBackend::Local {
            Some(LlamaBackend::init()?)
        } else {
            None
        };

        let local_model = if backend == LlmBackend::Local {
            if let Some(path) = model_path {
                let model = LlamaModel::load_from_file(
                    backend_handle.as_ref().unwrap(),
                    path,
                    &LlamaModelParams::default(),
                )?;
                Some(model)
            } else {
                None
            }
        } else {
            None
        };

        let api_key = match backend {
            LlmBackend::DeepSeek => std::env::var("DEEPSEEK_API_KEY").ok(),
            LlmBackend::OpenRouter => std::env::var("OPENROUTER_API_KEY").ok(),
            LlmBackend::Local => None,
        };

        Ok(Self {
            backend,
            backend_handle,
            local_model,
            api_key,
        })
    }

    fn endpoint_url(&self) -> Option<&'static str> {
        match self.backend {
            LlmBackend::DeepSeek => Some("https://api.deepseek.com/v1/chat/completions"),
            LlmBackend::OpenRouter => Some("https://openrouter.ai/api/v1/chat/completions"),
            LlmBackend::Local => None,
        }
    }

    fn cloud_model(&self) -> String {
        match self.backend {
            LlmBackend::DeepSeek => std::env::var("DEEPSEEK_MODEL")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "deepseek-chat".to_string()),
            LlmBackend::OpenRouter => std::env::var("OPENROUTER_MODEL")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "deepseek/deepseek-chat".to_string()),
            LlmBackend::Local => String::new(),
        }
    }

    pub fn stream_cleanup<F: FnMut(String) + Send + 'static>(
        &mut self,
        prompt: &str,
        callback: F,
    ) -> Result<()> {
        if self.backend == LlmBackend::Local {
            self.stream_local(prompt, callback)
        } else {
            self.stream_cloud(prompt, callback)
        }
    }

    fn stream_local<F: FnMut(String) + Send + 'static>(
        &mut self,
        prompt: &str,
        mut callback: F,
    ) -> Result<()> {
        let backend = self
            .backend_handle
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Local LLM backend not initialized"))?;

        let model = self
            .local_model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Local LLM model not loaded"))?;

        let mut ctx = model.new_context(backend, LlamaContextParams::default())?;
        let mut batch = LlamaBatch::new(512, 1);

        let tokens = model.str_to_token(prompt, AddBos::Always)?;
        for (i, &tok) in tokens.iter().enumerate() {
            batch.add(tok, i as i32, &[0], i == tokens.len() - 1)?;
        }

        ctx.decode(&mut batch)?;

        let mut n_cur = tokens.len() as i32;
        let sampler = LlamaSampler::greedy();
        let eos_token = model.token_eos();
        let max_tokens = 512;

        loop {
            let token = sampler.sample(&ctx, n_cur);

            let piece = model.token_to_bytes(token, Special::Plaintext)?;
            let s = String::from_utf8_lossy(&piece).to_string();
            callback(s);

            batch.add(token, n_cur, &[0], n_cur == tokens.len() as i32)?;
            ctx.decode(&mut batch)?;
            n_cur += 1;

            if token == eos_token || n_cur >= max_tokens {
                break;
            }
        }

        Ok(())
    }

    fn stream_cloud<F: FnMut(String) + Send + 'static>(
        &mut self,
        prompt: &str,
        mut callback: F,
    ) -> Result<()> {
        let api_key = self
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("API key not set"))?;

        let url = self
            .endpoint_url()
            .ok_or_else(|| anyhow::anyhow!("Invalid backend"))?;

        let body = serde_json::json!({
            "model": self.cloud_model(),
            "stream": true,
            "messages": [{"role": "user", "content": prompt}]
        });

        let api_key_owned = api_key.clone();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        runtime.block_on(async {
            let client = reqwest::Client::new();
            let response = client
                .post(url)
                .header("Authorization", format!("Bearer {}", api_key_owned))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("HTTP request failed: {}", e))?;

            let mut stream = response.bytes_stream();
            use futures_util::StreamExt;
            let mut buffer = String::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| anyhow::anyhow!("Stream error: {}", e))?;
                let text = String::from_utf8_lossy(&chunk);
                for token in drain_sse_buffer(&mut buffer, &text) {
                    callback(token);
                }
            }
            // Flush any trailing line without a newline terminator.
            for token in drain_sse_buffer(&mut buffer, "\n") {
                callback(token);
            }

            Ok(())
        })
    }
}

/// Extract the delta text from a single SSE line.
///
/// Handles OpenAI-compatible chunks: strips an optional `data:` prefix,
/// skips `[DONE]` sentinels / empty lines / SSE comments, and returns
/// `choices[0].delta.content` when present. Returns `None` when the line
/// carries no token text.
fn extract_delta_content(line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    // SSE comments (keep-alive `: ...`) carry no data.
    if line.starts_with(':') {
        return None;
    }
    let data = if let Some(rest) = line.strip_prefix("data:") {
        rest.trim()
    } else if line.starts_with('{') {
        // Lenient: accept bare JSON lines (some proxies strip the prefix).
        line
    } else {
        return None;
    };
    if data.is_empty() || data == "[DONE]" {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    value
        .get("choices")?
        .get(0)?
        .get("delta")?
        .get("content")?
        .as_str()
        .map(|s| s.to_string())
}

/// Append `chunk_text` to `buffer`, extract complete `\n`-terminated lines,
/// and return the parsed token texts. Any incomplete trailing line stays in
/// `buffer` for the next chunk.
fn drain_sse_buffer(buffer: &mut String, chunk_text: &str) -> Vec<String> {
    buffer.push_str(chunk_text);
    let mut tokens = Vec::new();
    while let Some(pos) = buffer.find('\n') {
        let line: String = buffer.drain(..=pos).collect();
        if let Some(token) = extract_delta_content(&line) {
            if !token.is_empty() {
                tokens.push(token);
            }
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_delta_content_from_data_line() {
        let line = r#"data: {"choices":[{"delta":{"content":"hello"}}]}"#;
        assert_eq!(extract_delta_content(line).as_deref(), Some("hello"));
    }

    #[test]
    fn skips_done_empty_and_comment_lines() {
        assert_eq!(extract_delta_content("data: [DONE]"), None);
        assert_eq!(extract_delta_content(""), None);
        assert_eq!(extract_delta_content(": keep-alive"), None);
        assert_eq!(
            extract_delta_content(r#"data: {"choices":[{"delta":{}}]}"#),
            None
        );
    }

    #[test]
    fn buffers_incomplete_lines_across_chunks() {
        let mut buffer = String::new();
        let first = drain_sse_buffer(
            &mut buffer,
            "data: {\"choices\":[{\"delta\":{\"content\":\"hel",
        );
        assert!(first.is_empty());
        assert!(!buffer.is_empty());
        let second = drain_sse_buffer(
            &mut buffer,
            "lo\"}}]}\n\ndata: [DONE]\n",
        );
        assert_eq!(second, vec!["hello".to_string()]);
        assert!(buffer.is_empty());
    }
}
