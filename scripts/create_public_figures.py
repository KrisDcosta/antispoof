"""Create polished public figures from committed metric summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"


COLORS = {
    "lfcc": "#5B7C99",
    "lcnn": "#2B6CB0",
    "aasist": "#7A5195",
    "wavlm": "#2F855A",
    "fusion": "#C05621",
    "ref": "#8A8F98",
}


def load_json(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def percent(value: float) -> float:
    return 100.0 * float(value)


def savefig(name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(path.relative_to(ROOT))


def eval_eer(summary: dict) -> float:
    return percent(summary["splits"]["eval"]["eer"])


def dev_eer(summary: dict) -> float:
    return percent(summary["splits"]["dev"]["eer"])


def plot_eval_comparison() -> None:
    lcnn = load_json("results/neural/metrics/lcnn_logmel_full_seed42_30ep_summary.json")
    aasist = load_json("results/neural/metrics/aasist_lite_waveform_seed42_100ep_summary.json")
    wavlm = load_json("results/neural/metrics/ssl_wavlm_pooled_full_seed42_50ep_summary.json")
    fusion = load_json("results/fusion/metrics/lcnn_wavlm_score_fusion_seed42_summary.json")
    lfcc = load_json("results/baseline/metrics/gmm_lfcc_64c_diag_std_300000frames_seed42_eval_metrics.json")

    labels = [
        "Fusion\nLCNN+WavLM",
        "WavLM\npooled MLP",
        "LCNN\nlog-mel",
        "Official\nLFCC-GMM",
        "Official\nCQCC-GMM",
        "LFCC\nGMM-LLR",
        "AASIST-lite\nwaveform",
    ]
    values = [
        percent(fusion["selected_eval_eer"]),
        eval_eer(wavlm),
        eval_eer(lcnn),
        8.09,
        9.57,
        percent(lfcc["metrics"]["eer"]),
        eval_eer(aasist),
    ]
    colors = [
        COLORS["fusion"],
        COLORS["wavlm"],
        COLORS["lcnn"],
        COLORS["ref"],
        COLORS["ref"],
        COLORS["lfcc"],
        COLORS["aasist"],
    ]

    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bars = ax.bar(labels, values, color=colors, edgecolor="#1F2933", linewidth=0.6)
    ax.set_title("ASVspoof 2019 LA Eval EER by System", fontsize=15, weight="bold", pad=16)
    ax.set_ylabel("Eval EER (%)")
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )
    ax.text(
        0.01,
        -0.20,
        "Lower is better. WavLM and fusion are external-pretrained/applied; LCNN, AASIST-lite, and GMM systems are protocol-comparable.",
        transform=ax.transAxes,
        fontsize=9,
        color="#52606D",
    )
    fig.tight_layout()
    savefig("eval_eer_comparison.png")


def plot_per_attack() -> None:
    lcnn = load_json("results/neural/metrics/lcnn_logmel_full_seed42_30ep_summary.json")
    wavlm = load_json("results/neural/metrics/ssl_wavlm_pooled_full_seed42_50ep_summary.json")
    fusion = load_json("results/fusion/metrics/lcnn_wavlm_score_fusion_seed42_summary.json")

    attacks = [f"A{i:02d}" for i in range(7, 20)]
    series = {
        "LCNN": [percent(lcnn["splits"]["eval"]["per_attack_eer"][a]) for a in attacks],
        "WavLM": [percent(wavlm["splits"]["eval"]["per_attack_eer"][a]) for a in attacks],
        "Fusion": [percent(fusion["selected_eval_per_attack_eer"][a]) for a in attacks],
    }

    x = np.arange(len(attacks))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(x - width, series["LCNN"], width, label="LCNN", color=COLORS["lcnn"])
    ax.bar(x, series["WavLM"], width, label="WavLM", color=COLORS["wavlm"])
    ax.bar(x + width, series["Fusion"], width, label="Fusion", color=COLORS["fusion"])
    ax.set_title("Eval Per-Attack EER: Complementary Error Profiles", fontsize=15, weight="bold", pad=16)
    ax.set_ylabel("EER (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    ax.text(
        0.01,
        -0.18,
        "Fusion reduces the dominant LCNN failures on A17/A18 while preserving stronger LCNN behavior on A19.",
        transform=ax.transAxes,
        fontsize=9,
        color="#52606D",
    )
    fig.tight_layout()
    savefig("per_attack_eval_eer.png")


def plot_robustness() -> None:
    robust = load_json("results/robustness/metrics/phase5_eval_corruptions_summary.json")
    conditions = [
        ("clean", "Clean"),
        ("clip_0p4", "Clip\n0.4"),
        ("resample_8k", "Resample\n8 kHz"),
        ("noise_20db", "Noise\n20 dB"),
        ("noise_10db", "Noise\n10 dB"),
        ("noise_5db", "Noise\n5 dB"),
    ]
    systems = [("lcnn", "LCNN", COLORS["lcnn"]), ("wavlm", "WavLM", COLORS["wavlm"]), ("fusion", "Fusion", COLORS["fusion"])]
    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for key, label, color in systems:
        values = [percent(robust["primary"][f"{key}::{condition}"]["eer"]) for condition, _ in conditions]
        ax.plot(x, values, marker="o", linewidth=2.4, markersize=6, label=label, color=color)
        for xi, value in zip(x, values):
            condition_name = conditions[int(xi)][0]
            if condition_name in {"clean", "noise_10db"}:
                ax.text(xi, value + 1.0, f"{value:.1f}", ha="center", fontsize=8, color=color)
    ax.set_title("Robustness EER Under Deterministic Audio Corruptions", fontsize=15, weight="bold", pad=16)
    ax.set_ylabel("Eval EER (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in conditions])
    ax.set_ylim(0, 46)
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    ax.text(
        0.01,
        -0.18,
        "Fusion is strongest on clean audio; WavLM is more stable under additive noise and heavier channel shifts.",
        transform=ax.transAxes,
        fontsize=9,
        color="#52606D",
    )
    fig.tight_layout()
    savefig("robustness_eer_by_condition.png")


def plot_fusion_alpha() -> None:
    fusion = load_json("results/fusion/metrics/lcnn_wavlm_score_fusion_seed42_summary.json")
    candidates = fusion["methods"]["weighted_mean"]["rule"]["alpha_candidates"]
    alphas = [c["alpha"] for c in candidates]
    dev_eers = [percent(c["dev_eer"]) for c in candidates]
    selected_alpha = fusion["methods"]["weighted_mean"]["rule"]["alpha"]
    selected_eer = percent(fusion["selected_dev_eer"])

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(alphas, dev_eers, color=COLORS["fusion"], marker="o", linewidth=2.4)
    ax.axvline(selected_alpha, color="#1F2933", linestyle="--", linewidth=1.2)
    ax.scatter([selected_alpha], [selected_eer], s=90, color=COLORS["fusion"], edgecolor="#1F2933", zorder=3)
    ax.text(selected_alpha + 0.02, selected_eer + 0.05, "selected alpha=0.7", fontsize=9, weight="bold")
    ax.set_title("Dev-Selected LCNN+WavLM Fusion Weight", fontsize=15, weight="bold", pad=16)
    ax.set_xlabel("LCNN weight alpha")
    ax.set_ylabel("Dev EER (%)")
    ax.set_xticks(alphas)
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        -0.20,
        "Fusion selection uses dev only; the frozen alpha=0.7 rule is then applied to eval.",
        transform=ax.transAxes,
        fontsize=9,
        color="#52606D",
    )
    fig.tight_layout()
    savefig("fusion_alpha_sweep.png")


def add_box(ax, xy: tuple[float, float], text: str, color: str, width: float = 2.4, height: float = 0.78) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=1.0,
        edgecolor="#1F2933",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9, weight="bold", color="#111827")


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#52606D")
    ax.add_patch(arrow)


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    add_box(ax, (0.4, 3.1), "ASVspoof 2019 LA\naudio + protocols", "#E5EAF0", width=2.2, height=1.0)
    add_box(ax, (3.2, 5.4), "Cepstral features\nLFCC / MFCC / CQCC", "#D9EAF7", width=2.4)
    add_box(ax, (3.2, 3.8), "Log-mel\nspectrogram", "#D9EAF7", width=2.4)
    add_box(ax, (3.2, 2.2), "Raw waveform\ncrop/pad", "#D9EAF7", width=2.4)
    add_box(ax, (3.2, 0.6), "Frozen WavLM\nmean+std pooling", "#D9F0E3", width=2.4)

    add_box(ax, (6.1, 5.4), "Dual GMM\nLLR", "#FCE7D7", width=1.9)
    add_box(ax, (6.1, 3.8), "LCNN", "#FCE7D7", width=1.9)
    add_box(ax, (6.1, 2.2), "AASIST-lite", "#FCE7D7", width=1.9)
    add_box(ax, (6.1, 0.6), "MLP head", "#FCE7D7", width=1.9)

    add_box(ax, (8.6, 3.7), "Score CSVs\n+ metrics", "#E8E2F3", width=2.0)
    add_box(ax, (8.6, 1.9), "LCNN+WavLM\nscore fusion", "#F6D6C3", width=2.0)
    add_box(ax, (8.6, 0.25), "Robustness\nsweep", "#F6D6C3", width=2.0)
    add_box(ax, (8.6, 5.45), "Per-attack\nanalysis", "#E8E2F3", width=2.0)

    for y in [5.9, 4.3, 2.7, 1.1]:
        add_arrow(ax, (2.6, 3.6), (3.2, y))
    for y in [5.79, 4.19, 2.59, 0.99]:
        add_arrow(ax, (5.6, y), (6.1, y))
        add_arrow(ax, (8.0, y), (8.6, 4.1 if y > 1.5 else 2.3))
    add_arrow(ax, (9.6, 3.7), (9.6, 2.68))
    add_arrow(ax, (9.6, 1.9), (9.6, 1.03))
    add_arrow(ax, (9.6, 4.48), (9.6, 5.45))

    ax.text(0.4, 7.25, "Speech Anti-Spoofing Evaluation Workflow", fontsize=16, weight="bold", color="#111827")
    ax.text(
        0.4,
        6.88,
        "Clean benchmark scoring, external-pretrained applied comparison, score fusion, and corruption robustness in one harness.",
        fontsize=10,
        color="#52606D",
    )
    savefig("system_architecture.png")


def main() -> None:
    plot_eval_comparison()
    plot_per_attack()
    plot_robustness()
    plot_fusion_alpha()
    plot_architecture()


if __name__ == "__main__":
    main()
