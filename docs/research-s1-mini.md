# Research: S1-Mini / SuperWhisper

> Date: 2026-09-09

## 1. What Is It?

**S1-mini** is a **text normalizer** for speech-to-text output, made by **Superwhisper** (https://superwhisper.com). It is **not** a speech-to-text model itself — it does not listen to audio.

**Architecture:** A causal language model (decoder-only), fine-tuned from **Qwen/Qwen3-0.6B** (596M unique parameters, 28 layers, GQA with 16 Q heads / 8 KV heads, hidden_size 1024, vocab_size 151936).

**What it does:** Takes a raw ASR transcript (from Whisper, Parakeet, etc.) and cleans it up:
- Removes filler words (`um`, `uh`)
- Resolves false starts / self-corrections (e.g., `"friday no wait make that thursday"` → `Thursday`)
- Applies punctuation and capitalization
- Renders spoken numbers, dates, times, currency, and email addresses in written form
- Controlled via a 3-axis "control line": Styling, Structure, Context

**Pipeline position:**
```
audio → ASR (Whisper, Parakeet, …) → S1-mini → clean text
```

**Release:** v1, August 2026. Open weights under **Apache 2.0 + naming clause**.

**Model card:** https://huggingface.co/superwhisper/s1-mini
**GGUF builds:** https://huggingface.co/superwhisper/s1-mini-GGUF
**Blog:** https://superwhisper.com/blog/s1

## 2. Availability in sherpa-onnx

**S1-mini is NOT available in sherpa-onnx**, and it **cannot be used** with sherpa-onnx in its current form.

**Why:** sherpa-onnx supports speech recognition models (encoder-decoder transducer, CTC, paraformer, Whisper encoder-decoder, etc.) that convert audio waveforms into text tokens. S1-mini is a decoder-only causal LLM that takes **text input** and produces **text output** — it sits downstream of ASR. sherpa-onnx has no runtime for arbitrary decoder-only LLMs / text generation models.

S1-mini does not appear in:
- The sherpa-onnx ASR models release page (500 assets, no s1-mini)
- The sherpa-onnx pretrained models documentation
- Any sherpa-onnx GitHub issues or discussions

**Available runtimes for S1-mini:**
| Runtime | Format | How |
|---------|--------|-----|
| Transformers (Python) | SafeTensors (BF16) | `pip install transformers` |
| vLLM | SafeTensors | `vllm serve superwhisper/s1-mini` |
| SGLang | SafeTensors | `sglang.launch_server` |
| llama.cpp | GGUF | `llama-server -hf superwhisper/s1-mini-GGUF:Q4_K_M` |
| Ollama | GGUF | `ollama run hf.co/superwhisper/s1-mini-GGUF:F16` |
| LM Studio | GGUF | Load from HuggingFace |
| ONNX Runtime GenAI | ONNX (unofficial) | `elbruno/s1-mini-onnx` |

## 3. Available Downloads

### Official (SafeTensors)
- **Repo:** `superwhisper/s1-mini`
- **File:** `model.safetensors` — **1.5 GB** (BF16)
- **Total repo size:** ~1.52 GB

### Official GGUF (recommended for local/edge)
- **Repo:** `superwhisper/s1-mini-GGUF`
- **File:** `s1-mini-q4_k_m.gguf` — **462 MB** (recommended; the published accuracy was measured on this build)
- **File:** `s1-mini-f16.gguf` — **1.4 GB** (unquantized intermediate)
- **Ollama:** `ollama run hf.co/superwhisper/s1-mini-GGUF:Q4_K_M`

### Unofficial ONNX (third-party, for .NET / ONNX Runtime GenAI)
- **Repo:** `elbruno/s1-mini-onnx` (not endorsed by Superwhisper)
- **INT4 folder:** ~390 MB on disk
  - `model.onnx` — 299 KB
  - `model.onnx.data` — 394 MB
  - `tokenizer.json` — 11.4 MB
  - `tokenizer_config.json` — 723 B
  - `genai_config.json` — 1.57 KB
  - `chat_template.jinja` — 4.26 KB
- **FP16 folder:** ~1.2 GB (BROKEN on CPU with onnxruntime-genai 0.15.1 — GQA Reshape shape-mismatch bug)

**⚠️ The ONNX conversion is unofficial and uses a different file layout than what sherpa-onnx expects (see section 4).**

## 4. File Layout

### SafeTensors (original)
Standard HuggingFace Transformers layout:
```
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
merges.txt
vocab.json
chat_template.jinja
generation_config.json
```

### GGUF (official)
Single-file GGUF format for llama.cpp ecosystem:
```
s1-mini-q4_k_m.gguf    (462 MB, recommended)
s1-mini-f16.gguf        (1.4 GB)
```

### ONNX (unofficial, ONNX Runtime GenAI format)
```
int4/
  model.onnx           (299 KB — the graph)
  model.onnx.data      (394 MB — external weights)
  tokenizer.json
  tokenizer_config.json
  genai_config.json
  chat_template.jinja
```

**Critical: This is NOT the sherpa-onnx ONNX layout.** sherpa-onnx ONNX models use separate `encoder.onnx`, `decoder.onnx`, `joiner.onnx` files (for transducer), or `model.onnx` with `tokens.txt` (for Whisper/CTC). S1-mini's ONNX is a single monolithic decoder-only graph designed for ONNX Runtime GenAI's text-generation API.

## 5. Streaming vs Non-Streaming

**Non-streaming (offline), text-in / text-out only.**

S1-mini is not an ASR model — it doesn't process audio at all. It takes a complete text transcript and rewrites it. There is no streaming variant and no concept of incremental audio processing. The recommended input is up to ~1,000 tokens; longer transcripts should be chunked at sentence boundaries.

## 6. Language Support

**English only** (v1).

The model was trained and evaluated exclusively on English ASR transcripts. Evaluation: 94.8% token accuracy on a held-out set of 7,519 English cases (measured greedy on the Q4_K_M GGUF build).

## Summary

| Property | Value |
|----------|-------|
| **Type** | Text normalizer (post-processor for ASR) |
| **NOT** | A speech-to-text / ASR model |
| **Maker** | Superwhisper |
| **Architecture** | Qwen3-0.6B (decoder-only causal LM) |
| **Parameters** | 596M unique (0.6B) |
| **In sherpa-onnx?** | **No** — incompatible architecture |
| **Streaming?** | No — offline text-in / text-out |
| **Languages** | English only (v1) |
| **Best local format** | GGUF Q4_K_M (462 MB) |
| **License** | Apache 2.0 + naming clause |
| **Evaluation** | 94.8% token accuracy (7,519 English cases) |
