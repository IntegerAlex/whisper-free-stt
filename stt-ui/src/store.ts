// ── Global state store (Zustand-lite pattern with React Context) ──
import { createContext, useContext } from "react";

export type ASRBackend = "sherpa_onnx";
export type ASRMODEL = "parakeet" | "whisper-turbo" | "whisper-base";
export type LLMBackend = "local" | "deepseek" | "openrouter";
export type LLMMode = "off" | "cleanup" | "bullet_list" | "email" | "commit_message";

export type LlmModelBackend = "llama_cpp" | "deepseek" | "openrouter";

export interface LlmModelInfo {
  id: string;
  name: string;
  size: string;
  sizeBytes: number;
  bestFor: string;
  backend: LlmModelBackend;
  downloaded: boolean;
  recommended: boolean;
  url: string;
  filename?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  size: string;
  sizeBytes: number;
  speed: string;
  accuracy: string;
  bestFor: string;
  backend: ASRBackend;
  profile: ASRMODEL;
  downloaded: boolean;
  recommended: boolean;
  url: string;
}

export const MODEL_CATALOG: ModelInfo[] = [
  {
    id: "parakeet-tdt-0.6b-v2-int8",
    name: "Parakeet TDT 0.6B v2 (int8)",
    size: "~460 MB",
    sizeBytes: 482_468_385,
    speed: "🚀 Fastest",
    accuracy: "⭐⭐⭐⭐",
    bestFor: "Fast dictation (English)",
    backend: "sherpa_onnx",
    profile: "parakeet",
    downloaded: false,
    recommended: true,
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
  },
  {
    id: "whisper-large-v3-turbo-q5_1",
    name: "Whisper large-v3-turbo (Q5_1)",
    size: "~540 MB",
    sizeBytes: 563_790_207,
    speed: "⚡ Medium",
    accuracy: "⭐⭐⭐⭐⭐",
    bestFor: "Multilingual, high accuracy",
    backend: "sherpa_onnx",
    profile: "whisper-turbo",
    downloaded: false,
    recommended: false,
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2",
  },
  {
    id: "whisper-base-q5_1",
    name: "Whisper base (Q5_1)",
    size: "~200 MB",
    sizeBytes: 207_557_382,
    speed: "🚀 Fast",
    accuracy: "⭐⭐⭐",
    bestFor: "Lightweight, any language",
    backend: "sherpa_onnx",
    profile: "whisper-base",
    downloaded: false,
    recommended: false,
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.tar.bz2",
  },
];

export const LLM_MODEL_CATALOG: LlmModelInfo[] = [
  {
    id: "s1-mini-q4_k_m",
    name: "S1-Mini (Q4_K_M)",
    size: "~462 MB",
    sizeBytes: 484_219_808,
    bestFor: "ASR transcript cleanup",
    backend: "llama_cpp",
    downloaded: false,
    recommended: true,
    url: "https://huggingface.co/superwhisper/s1-mini-GGUF/resolve/main/s1-mini-q4_k_m.gguf",
    filename: "s1-mini-q4_k_m.gguf",
  },
  {
    id: "gemma-3-1b-it-q4_k_m",
    name: "Gemma 3 1B IT (Q4_K_M)",
    size: "~806 MB",
    sizeBytes: 806_058_272,
    bestFor: "Offline text cleaning",
    backend: "llama_cpp",
    downloaded: false,
    recommended: false,
    url: "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf",
    filename: "gemma-3-1b-it-q4_k_m.gguf",
  },
];

export interface SystemCheck {
  name: string;
  status: "pass" | "fail" | "pending" | "warning";
  message: string;
  fixHint?: string;
}

export interface OnboardingState {
  step: number;
  totalSteps: number;
  completed: boolean;
  skipped: boolean;
  systemChecks: SystemCheck[];
  selectedMicIndex: number | null;
  micLevel: number;
  clipboardEnabled: boolean;
  typingEnabled: boolean;
  preferredModel: string;
  preferredLLMBackend: LLMBackend;
  llmMode: LLMMode;
  modelDownloadProgress: Record<string, { percent: number; bytesDownloaded: number; bytesTotal: number; status: "idle" | "downloading" | "done" | "error" }>;
  error: string | null;
}

export type OnboardingAction =
  | { type: "SET_STEP"; step: number }
  | { type: "NEXT_STEP" }
  | { type: "SET_SYSTEM_CHECKS"; checks: SystemCheck[] }
  | { type: "SET_COMPLETED" }
  | { type: "SET_SKIPPED" }
  | { type: "SET_MIC"; index: number | null; level: number }
  | { type: "SET_CLIPBOARD"; enabled: boolean }
  | { type: "SET_TYPING"; enabled: boolean }
  | { type: "SET_MODEL"; name: string }
  | { type: "SET_LLM_BACKEND"; backend: LLMBackend }
  | { type: "SET_LLM_MODE"; mode: LLMMode }
  | { type: "SET_DOWNLOAD_PROGRESS"; name: string; percent: number; bytesDownloaded: number; bytesTotal: number; status: "idle" | "downloading" | "done" | "error" }
  | { type: "SET_ERROR"; error: string }
  | { type: "CLEAR_ERROR" };

export function onboardingReducer(state: OnboardingState, action: OnboardingAction): OnboardingState {
  switch (action.type) {
    case "SET_STEP":
      return { ...state, step: action.step };
    case "NEXT_STEP":
      return { ...state, step: Math.min(state.step + 1, state.totalSteps) };
    case "SET_SYSTEM_CHECKS":
      return { ...state, systemChecks: action.checks };
    case "SET_COMPLETED":
      return { ...state, completed: true };
    case "SET_SKIPPED":
      return { ...state, skipped: true, completed: true };
    case "SET_MIC":
      return { ...state, selectedMicIndex: action.index, micLevel: action.level };
    case "SET_CLIPBOARD":
      return { ...state, clipboardEnabled: action.enabled };
    case "SET_TYPING":
      return { ...state, typingEnabled: action.enabled };
    case "SET_MODEL":
      return { ...state, preferredModel: action.name };
    case "SET_LLM_BACKEND":
      return { ...state, preferredLLMBackend: action.backend };
    case "SET_LLM_MODE":
      return { ...state, llmMode: action.mode };
    case "SET_DOWNLOAD_PROGRESS":
      return {
        ...state,
        modelDownloadProgress: {
          ...state.modelDownloadProgress,
          [action.name]: {
            percent: action.percent,
            bytesDownloaded: action.bytesDownloaded,
            bytesTotal: action.bytesTotal,
            status: action.status,
          },
        },
      };
    case "SET_ERROR":
      return { ...state, error: action.error };
    case "CLEAR_ERROR":
      return { ...state, error: null };
    default:
      return state;
  }
}

export const DEFAULT_ONBOARDING: OnboardingState = {
  step: 0,
  totalSteps: 5,
  completed: false,
  skipped: false,
  systemChecks: [],
  selectedMicIndex: null,
  micLevel: 0,
  clipboardEnabled: true,
  typingEnabled: true,
  preferredModel: "parakeet-tdt-0.6b-v2-int8",
  preferredLLMBackend: "local",
  llmMode: "cleanup",
  modelDownloadProgress: {},
  error: null,
};

export type AppView = "onboarding" | "main";

export const AppStateContext = createContext<{
  onboarding: OnboardingState;
  onboardingDispatch: React.Dispatch<OnboardingAction>;
  view: AppView;
  setView: (v: AppView) => void;
} | null>(null);

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used within AppStateProvider");
  return ctx;
}
