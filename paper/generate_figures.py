#!/usr/bin/env python3
"""
Generate evaluation figures for the EMNLP 2026 paper.
Produces: latency comparison, VAD adaptation, dictionary correction progression.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

COLORS = {
    'primary': '#2196F3',
    'secondary': '#FF9800',
    'accent': '#4CAF50',
    'error': '#F44336',
    'bg': '#F5F5F5',
    'text': '#212121',
}


def plot_latency_comparison():
    """Figure 1: ASR latency from actual measured data (Table 3/4)."""
    # Data from Table 3/4 in paper
    profiles = ['speed\n(tiny.en)\n[CPU]', 'balanced\n(base.en)\n[CPU]', 
                'accuracy\n(small.en)\n[CPU]', 'small-cuda\n(small.en)\n[GPU]',
                'distil\n(distil-large-v3)\n[GPU]', 'turbo\n(large-v3-turbo)\n[GPU]']
    
    # Only balanced (base.en) and turbo (large-v3-turbo) are measured
    # Others are estimated from upstream benchmarks (marked with hatching)
    p50 = [None, 1.90, None, None, None, 0.77]
    p95 = [None, 2.33, None, None, None, 1.46]
    
    # Upstream estimates (from model cards, not our measurements)
    p50_est = [0.12, None, 0.45, 0.04, 0.025, None]
    p95_est = [0.22, None, 0.82, 0.08, 0.05, None]
    
    backends = ['CPU', 'CPU', 'CPU', 'GPU', 'GPU', 'GPU']
    cpu_color = COLORS['primary']
    gpu_color = COLORS['accent']
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(profiles))
    width = 0.35
    
    # Measured bars (solid)
    measured_p50 = [0 if v is None else v for v in p50]
    measured_p95 = [0 if v is None else v for v in p95]
    measured_color = [cpu_color if b == 'CPU' else gpu_color for b in backends]
    
    bars_p50 = ax.bar(x - width/2, measured_p50, width, label='P50 (measured)', 
                      color=measured_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars_p95 = ax.bar(x + width/2, measured_p95, width, label='P95 (measured)', 
                      color=measured_color, alpha=0.5, edgecolor='black', linewidth=0.5)
    
    # Estimated bars (hatched, transparent)
    est_p50 = [v if v is not None else 0 for v in p50_est]
    est_p95 = [v if v is not None else 0 for v in p95_est]
    
    bars_p50_est = ax.bar(x - width/2, est_p50, width, 
                          color=measured_color, alpha=0.2, hatch='///', edgecolor='gray')
    bars_p95_est = ax.bar(x + width/2, est_p95, width,
                          color=measured_color, alpha=0.1, hatch='///', edgecolor='gray')
    
    ax.set_ylabel('Latency (seconds)', fontsize=11)
    ax.set_xlabel('ASR Profile', fontsize=11)
    ax.set_title('ASR Latency Across Profiles (3s audio clip)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, fontsize=9)
    ax.set_ylim(0, 2.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    cpu_patch = Patch(facecolor=cpu_color, alpha=0.8, label='CPU (measured)', edgecolor='black')
    gpu_patch = Patch(facecolor=gpu_color, alpha=0.8, label='GPU (measured)', edgecolor='black')
    cpu_est_patch = Patch(facecolor=cpu_color, alpha=0.2, hatch='///', label='CPU (upstream est.)', edgecolor='gray')
    gpu_est_patch = Patch(facecolor=gpu_color, alpha=0.2, hatch='///', label='GPU (upstream est.)', edgecolor='gray')
    p50_patch = Patch(facecolor='white', edgecolor='black', label='P50', linewidth=1)
    p95_patch = Patch(facecolor='white', edgecolor='black', alpha=0.5, label='P95', linewidth=1)
    
    ax.legend(handles=[cpu_patch, gpu_patch, cpu_est_patch, gpu_est_patch, p50_patch, p95_patch],
              loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'latency_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'latency_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'latency_comparison.pdf'}")


def plot_vad_ablation():
    """Figure 2: VAD ablation with detection rate and FPR."""
    variants = ['Energy-only', 'Full composite\n(w=0.4)', 'High spectral\n(w=0.8)']
    conditions = ['Clean\n(SNR~20dB)', 'Moderate\n(SNR~10dB)', 'Noisy\n(SNR~5dB)']
    
    det_rates = np.array([
        [35, 70, 70],   # Energy-only
        [50, 40, 15],   # Full composite
        [55, 40, 40],   # High spectral
    ])
    
    fpr_rates = np.array([
        [12, 28, 35],   # Energy-only
        [5, 8, 3],      # Full composite
        [8, 10, 12],    # High spectral
    ])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    x = np.arange(len(variants))
    width = 0.2
    
    # Detection rate
    for i, cond in enumerate(conditions):
        bars = ax1.bar(x + i*width - width, det_rates[:, i], width, 
                       label=cond, alpha=0.8, color=plt.cm.Set2(i))
        for bar, val in zip(bars, det_rates[:, i]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val}%', ha='center', fontsize=8)
    
    ax1.set_ylabel('Detection Rate (%)', fontsize=11)
    ax1.set_xlabel('VAD Variant', fontsize=11)
    ax1.set_title('Detection Rate by Noise Condition', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=9)
    ax1.legend(fontsize=8, title='Condition')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # FPR
    for i, cond in enumerate(conditions):
        bars = ax2.bar(x + i*width - width, fpr_rates[:, i], width,
                       label=cond, alpha=0.8, color=plt.cm.Set2(i))
        for bar, val in zip(bars, fpr_rates[:, i]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}%', ha='center', fontsize=8)
    
    ax2.set_ylabel('False Positive Rate (%)', fontsize=11)
    ax2.set_xlabel('VAD Variant', fontsize=11)
    ax2.set_title('False Positive Rate by Noise Condition', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, fontsize=9)
    ax2.legend(fontsize=8, title='Condition')
    ax2.set_ylim(0, 45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'vad_ablation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'vad_ablation.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'vad_ablation.pdf'}")


def plot_dictionary_correction():
    """Figure 3: Dictionary correction progression across layers."""
    layers = ['Layer 1:\nExact regex', 'Layer 2:\nFuzzy phonetic', 'Layer 3:\nLLM context']
    # From the methodology: 4/5 -> 5/5 -> 5/5 (progressive cumulative correction)
    accuracy = [80.0, 100.0, 100.0]
    colors = [COLORS['secondary'], COLORS['primary'], COLORS['accent']]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(layers, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.0f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Cumulative Correction Accuracy (%)', fontsize=11)
    ax.set_title('Dictionary Correction Progression Across Layers', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    
    # Add note
    ax.text(0.5, -0.15, 'Test set: 5 medical/tech terms, 1 speaker, n=5 utterances/term',
            ha='center', va='top', transform=ax.transAxes, fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dictionary_correction.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'dictionary_correction.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'dictionary_correction.pdf'}")


def plot_vad_adaptation():
    """Figure 4: VAD noise floor adaptation (kept from original)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: Noise floor tracking
    time_s = np.linspace(0, 60, 6000)
    noise_slow = np.zeros_like(time_s)
    noise_fast = np.zeros_like(time_s)
    actual_noise = np.zeros_like(time_s)

    for i, t in enumerate(time_s):
        if 10 < t < 30 or t > 45:
            actual_noise[i] = 0.015
        else:
            actual_noise[i] = 0.005

        if i == 0:
            noise_slow[i] = 0.005
            noise_fast[i] = 0.005
        else:
            noise_slow[i] = 0.9999 * noise_slow[i-1] + 0.0001 * actual_noise[i]
            noise_fast[i] = 0.995 * noise_fast[i-1] + 0.005 * actual_noise[i]

    ax1.plot(time_s, actual_noise * 1000, 'k--', alpha=0.5, label='Actual noise', linewidth=1)
    ax1.plot(time_s, noise_slow * 1000, color=COLORS['primary'], label='Slow EMA (30min)', linewidth=2)
    ax1.plot(time_s, noise_fast * 1000, color=COLORS['secondary'], label='Fast EMA (2s)', linewidth=2)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Noise Floor (RMS × 1000)', fontsize=11)
    ax1.set_title('Dual-Timescale Noise Tracking', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Right: Spectral features + hysteresis
    t = np.linspace(0, 2, 200)
    speech = np.zeros_like(t)
    speech[50:80] = 0.3 * np.sin(2 * np.pi * 200 * t[50:80])

    energy = np.abs(speech) + 0.005
    snr_db = 20 * np.log10(energy / 0.005 + 1e-10)
    composite = np.clip(snr_db / 6.0, 0, 2)

    ax2.plot(t, composite, color=COLORS['accent'], linewidth=2, label='Composite score')
    ax2.axhline(y=1.67, color=COLORS['error'], linestyle='--', label='Onset (1.67)', alpha=0.7)
    ax2.axhline(y=0.50, color=COLORS['primary'], linestyle='--', label='Offset (0.50)', alpha=0.7)
    ax2.fill_between(t, 0, composite, alpha=0.2, color=COLORS['accent'])
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Speech Score', fontsize=11)
    ax2.set_title('Hysteresis VAD State Machine', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 2.5)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'vad_adaptation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'vad_adaptation.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'vad_adaptation.pdf'}")


if __name__ == "__main__":
    print("Generating paper figures...")
    plot_latency_comparison()
    plot_vad_ablation()
    plot_dictionary_correction()
    plot_vad_adaptation()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
