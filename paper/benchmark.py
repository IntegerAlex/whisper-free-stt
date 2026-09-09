#!/usr/bin/env python3
"""
Benchmark script for Floure paper evaluation.
Runs ASR latency, dictionary accuracy, and VAD robustness tests.
Outputs formatted tables for the EMNLP 2026 paper.
"""
import time
import json
import sys
import os
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_asr_latency(num_samples=5):
    """Benchmark ASR latency across profiles."""
    from stt.config import AppConfig, TranscriptionConfig, TranscriptionBackend
    from stt.transcription import transcribe
    import numpy as np

    profiles = [
        ("speed", TranscriptionBackend.WHISPER_CPP, "tiny.en", 1),
        ("balanced", TranscriptionBackend.WHISPER_CPP, "base.en", 1),
        ("accuracy", TranscriptionBackend.WHISPER_CPP, "small.en", 3),
    ]

    # Only add GPU profiles if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            profiles.extend([
                ("small-cuda", TranscriptionBackend.FASTER_WHISPER, "small.en", 3),
                ("distil", TranscriptionBackend.FASTER_WHISPER, "distil-large-v3", 5),
                ("turbo", TranscriptionBackend.FASTER_WHISPER, "large-v3-turbo", 5),
            ])
    except ImportError:
        pass

    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Simulate speech-like audio
    audio = (0.3 * np.sin(2 * np.pi * 200 * t) +
             0.2 * np.sin(2 * np.pi * 400 * t) +
             0.1 * np.sin(2 * np.pi * 600 * t))
    audio *= (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    audio = audio.astype(np.float32)

    results = {}
    for profile_name, backend, model_name, beam_size in profiles:
        print(f"\nBenchmarking {profile_name} ({backend.name}/{model_name})...")
        config = TranscriptionConfig(
            backend=backend,
            model_name=model_name,
            beam_size=beam_size,
        )

        latencies = []
        for i in range(num_samples):
            t0 = time.perf_counter()
            try:
                result = transcribe(audio, sr, config)
                t1 = time.perf_counter()
                if result and result.text:
                    latencies.append(t1 - t0)
                    print(f"  [{i+1}/{num_samples}] {t1-t0:.3f}s - {result.text[:50]}...")
                else:
                    print(f"  [{i+1}/{num_samples}] empty result")
            except Exception as e:
                print(f"  [{i+1}/{num_samples}] ERROR: {e}")

        if latencies:
            p50 = statistics.median(latencies)
            p95 = sorted(latencies)[int(len(latencies) * 0.95)]
            rtf = p50 / duration
            results[profile_name] = {
                "backend": backend.name,
                "model": model_name,
                "n": len(latencies),
                "p50": round(p50, 3),
                "p95": round(p95, 3),
                "rtf": round(rtf, 4),
                "mean": round(statistics.mean(latencies), 3),
            }
            print(f"  -> P50={p50:.3f}s P95={p95:.3f}s RTF={rtf:.4f}")

    return results


def benchmark_dictionary_accuracy():
    """Benchmark 3-layer dictionary correction using HistoryStore."""
    from stt.history import get_store
    import tempfile
    import os

    store = None
    try:
        # Create a temporary store to avoid polluting main DB
        tmp = tempfile.mktemp(suffix='.db')
        store = get_store(tmp)

        # Add test dictionary entries
        test_entries = [
            ("hipe tension", "hypertension", "medical"),
            ("echocardiograam", "echocardiogram", "medical"),
            ("UIUX", "UI/UX", "tech"),
            ("gabapentin", "gabapentin", "medical"),
            ("metformin", "metformin", "medical"),
        ]

        for wrong, correct, category in test_entries:
            store.add_dictionary_entry(wrong, correct, category=category)

        # Test cases: (raw_text, expected_after_all_layers)
        test_cases = [
            ("the patient has hipe tension", "the patient has hypertension"),
            ("echocardiograam showed normal function", "echocardiogram showed normal function"),
            ("the UIUX design was great", "the UI/UX design was great"),
            ("gabapentin 300mg TID", "gabapentin 300mg TID"),
            ("metformin 500mg BID", "metformin 500mg BID"),
        ]

        layer1_correct = 0
        layer2_correct = 0
        total = len(test_cases)

        for raw, expected in test_cases:
            # Layer 1: exact regex
            l1 = store.apply_dictionary_replacements(raw)
            if l1 == expected:
                layer1_correct += 1

            # Layer 2: fuzzy
            l2 = store.apply_fuzzy_replacements(l1)
            if l2 == expected:
                layer2_correct += 1

        return {
            "total": total,
            "layer1_correct": layer1_correct,
            "layer2_correct": layer2_correct,
            "layer1_pct": round(layer1_correct / total * 100, 1),
            "cumulative_pct": round(layer2_correct / total * 100, 1),
        }
    finally:
        if store:
            try:
                store._conn.close()
            except Exception:
                pass
        # Clean up temp DB
        for suffix in ['', '-wal', '-shm']:
            try:
                os.unlink(tmp + suffix)
            except Exception:
                pass


def benchmark_vad_robustness():
    """Benchmark VAD under different acoustic conditions."""
    from stt.vad import StreamingEndpointDetector
    from stt.config import VADConfig
    import numpy as np

    config = VADConfig()
    sr = 16000
    block_size = 1024

    conditions = [
        ("quiet", 0.003),
        ("hvac", 0.010),
        ("noisy", 0.025),
    ]

    results = {}
    for condition_name, noise_level in conditions:
        detector = StreamingEndpointDetector(config, sr, block_size)

        duration = 30.0
        num_blocks = int(duration * sr / block_size)

        speech_starts = 0
        speech_ends = 0

        for i in range(num_blocks):
            t = np.arange(block_size) / sr + (i * block_size / sr)
            # Speech burst every 3 seconds
            if int(t[0]) % 3 == 0 and (t[0] % 3) < 0.5:
                signal = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
                signal *= (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
            else:
                signal = np.zeros(block_size, dtype=np.float32)

            noise = np.random.randn(block_size).astype(np.float32) * noise_level
            chunk = signal + noise

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            chunk_start = i * block_size
            chunk_end = chunk_start + block_size

            event = detector.update(rms, chunk_start, chunk_end, chunk)
            if event:
                if event.kind == "start":
                    speech_starts += 1
                elif event.kind == "end":
                    speech_ends += 1

        results[condition_name] = {
            "noise_level": noise_level,
            "speech_starts": speech_starts,
            "speech_ends": speech_ends,
        }

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("FLOURE BENCHMARK SUITE")
    print("=" * 60)

    all_results = {}

    # ASR Latency
    print("\n[1/3] ASR Latency Benchmark")
    print("-" * 40)
    try:
        asr_results = benchmark_asr_latency(num_samples=3)
        all_results["asr"] = asr_results
        if asr_results:
            print("\nASR Latency Table (LaTeX):")
            print("\\begin{tabular}{@{}lccc@{}}")
            print("\\toprule")
            print("Profile & P50 (s) & P95 (s) & RTF \\\\")
            print("\\midrule")
            for profile, data in asr_results.items():
                print(f"{profile} & {data['p50']:.3f} & {data['p95']:.3f} & {data['rtf']:.4f} \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")
    except Exception as e:
        print(f"ASR benchmark failed: {e}")
        import traceback; traceback.print_exc()
        all_results["asr"] = {}

    # Dictionary
    print("\n[2/3] Dictionary Accuracy Benchmark")
    print("-" * 40)
    try:
        dict_results = benchmark_dictionary_accuracy()
        all_results["dictionary"] = dict_results
        print(f"  Total cases: {dict_results['total']}")
        print(f"  Layer 1 (exact): {dict_results['layer1_correct']}/{dict_results['total']} = {dict_results['layer1_pct']}%")
        print(f"  Layer 1+2 (fuzzy): {dict_results['layer2_correct']}/{dict_results['total']} = {dict_results['cumulative_pct']}%")
    except Exception as e:
        print(f"Dictionary benchmark failed: {e}")
        import traceback; traceback.print_exc()
        all_results["dictionary"] = {}

    # VAD
    print("\n[3/3] VAD Robustness Benchmark")
    print("-" * 40)
    try:
        vad_results = benchmark_vad_robustness()
        all_results["vad"] = vad_results
        for condition, data in vad_results.items():
            print(f"  {condition}: noise={data['noise_level']:.3f} "
                  f"starts={data['speech_starts']} ends={data['speech_ends']}")
    except Exception as e:
        print(f"VAD benchmark failed: {e}")
        import traceback; traceback.print_exc()
        all_results["vad"] = {}

    # Save results
    all_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
