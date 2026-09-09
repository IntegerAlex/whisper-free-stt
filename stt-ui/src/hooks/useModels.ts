// ── Model management hook: check status, download, track progress ──
import { useState, useCallback, useEffect, useRef } from "react";
import { MODEL_CATALOG } from "../store";
import type { ASRBackend, ModelInfo } from "../store";

export interface ModelStatusEntry {
  name: string;
  backend: ASRBackend | "whisper_cpp" | "faster_whisper";
  downloaded: boolean;
  downloading: boolean;
  progress: number;
  error: string | null;
  sizeBytes: number;
  path: string;
}

interface RustModelStatus {
  id: string;
  name: string;
  downloaded: boolean;
  path: string;
  size_bytes: number;
}

function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export function useModels() {
  const [models, setModels] = useState<ModelStatusEntry[]>(() =>
    MODEL_CATALOG.map((m) => ({
      name: m.name,
      backend: m.backend,
      downloaded: false,
      downloading: false,
      progress: 0,
      error: null,
      sizeBytes: 0,
      path: "",
    }))
  );
  const [loading, setLoading] = useState(true);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refreshModels = useCallback(async () => {
    if (!isTauri()) {
      setLoading(false);
      return;
    }
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const rustStatuses = await invoke<RustModelStatus[]>("check_model_status");

      setModels((prev) =>
        prev.map((m) => {
          const status = rustStatuses.find((s) => s.name === m.name);
          if (status) {
            return {
              ...m,
              downloaded: status.downloaded,
              sizeBytes: status.size_bytes,
              path: status.path,
            };
          }
          return m;
        })
      );
    } catch (err) {
      setGlobalError(err instanceof Error ? err.message : "Failed to check model status");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshModels();
  }, [refreshModels]);

  const downloadModel = useCallback(async (modelName: string) => {
    if (!isTauri()) {
      setGlobalError("Model download is only available in the desktop app");
      return;
    }

    const entry = MODEL_CATALOG.find((m) => m.name === modelName);
    if (!entry) {
      setGlobalError(`Model "${modelName}" not found in catalog`);
      return;
    }

    setModels((prev) =>
      prev.map((m) =>
        m.name === modelName
          ? { ...m, downloading: true, progress: 0, error: null }
          : m
      )
    );

    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("download_model", { id: entry.id });

      // Poll check_model_status until the model appears downloaded.
      // download_model is fire-and-forget; polling is the only way to
      // know when it finishes.
      const poll = setInterval(async () => {
        try {
          const statuses = await invoke<RustModelStatus[]>("check_model_status");
          const status = statuses.find((s) => s.id === entry.id);
          if (status?.downloaded) {
            if (pollingRef.current) clearInterval(pollingRef.current);
            pollingRef.current = null;
            setModels((prev) =>
              prev.map((m) =>
                m.name === modelName
                  ? { ...m, downloading: false, progress: 100 }
                  : m
              )
            );
            await refreshModels();
          }
        } catch {
          /* retry next tick */
        }
      }, 2000);

      pollingRef.current = poll;
    } catch (err) {
      setModels((prev) =>
        prev.map((m) =>
          m.name === modelName
            ? { ...m, downloading: false, progress: 0, error: err instanceof Error ? err.message : "Download failed" }
            : m
        )
      );
    }
  }, [refreshModels]);

  const deleteModel = useCallback(async (modelName: string) => {
    if (!isTauri()) return;

    const model = models.find((m) => m.name === modelName);
    if (!model || !model.downloaded || !model.path) return;

    try {
      const { invoke: invokeCmd } = await import("@tauri-apps/api/core");
      await invokeCmd("delete_model_file", { path: model.path });
      await refreshModels();
    } catch {
      setGlobalError(`Failed to delete ${modelName}`);
    }
  }, [models, refreshModels]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  return {
    models,
    loading,
    globalError,
    refreshModels,
    downloadModel,
    deleteModel,
  };
}

export function getModelInfo(modelName: string): ModelInfo | undefined {
  return MODEL_CATALOG.find((m) => m.name === modelName);
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(i > 1 ? 1 : 0)} ${units[i]}`;
}
