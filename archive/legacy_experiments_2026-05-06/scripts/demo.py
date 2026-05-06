"""
Fix E: Gradio demo — upload a .flac/.wav file, get bonafide/spoof prediction.

Shows:
  - Prediction label + confidence
  - WCQCC feature heatmap
  - MFCC feature heatmap (for comparison)

Deploy free to HuggingFace Spaces: gradio deploy

Usage (local):
    python scripts/demo.py --model results/models/rf_wcqcc.pkl
"""

import argparse
import os
import sys
import io

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import mfcc_frames, wcqcc_frames, mfcc_vec, wcqcc_vec, SR


def predict(audio_path: str, model) -> tuple[str, float, str]:
    """Returns (label, confidence, explanation)."""
    audio, sr = librosa.load(audio_path, sr=SR, mono=True)
    vec = wcqcc_vec(audio, sr).reshape(1, -1)
    proba = model.predict_proba(vec)[0]   # [P(spoof), P(bonafide)]
    p_bonafide = float(proba[1])
    label = "Bonafide" if p_bonafide >= 0.5 else "Spoof"
    confidence = p_bonafide if p_bonafide >= 0.5 else 1.0 - p_bonafide
    explanation = (
        f"Model is {confidence*100:.1f}% confident this is {label} speech.\n"
        f"P(bonafide) = {p_bonafide:.3f} | P(spoof) = {1-p_bonafide:.3f}"
    )
    return label, confidence, explanation


def feature_plot(audio_path: str):
    """Returns a matplotlib figure with WCQCC and MFCC heatmaps."""
    audio, sr = librosa.load(audio_path, sr=SR, mono=True)
    wcqcc = wcqcc_frames(audio, sr).T   # [N_COEFF, T]
    mfcc  = mfcc_frames(audio, sr).T   # [N_COEFF, T]

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    for ax, feat, title in zip(axes, [wcqcc, mfcc], ["WCQCC", "MFCC"]):
        im = ax.imshow(feat, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Coefficient")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def launch_gradio(model):
    try:
        import gradio as gr
    except ImportError:
        print("Install gradio: pip install gradio")
        return

    def gradio_predict(audio_file):
        if audio_file is None:
            return "No file uploaded", None
        label, conf, explanation = predict(audio_file, model)
        fig = feature_plot(audio_file)

        color = "#2196F3" if label == "Bonafide" else "#F44336"
        result_html = (
            f"<div style='font-size:2em; color:{color}; font-weight:bold'>"
            f"{label}</div>"
            f"<div style='margin-top:8px'>{explanation}</div>"
        )
        return result_html, fig

    with gr.Blocks(title="Speech Anti-Spoofing Demo") as demo:
        gr.Markdown("## Speech Anti-Spoofing — ASVspoof 2019 LA\n"
                    "Upload a speech file (.wav or .flac). "
                    "Model detects whether it is **bonafide** (genuine) "
                    "or **spoof** (synthesized / voice-converted).")

        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="Upload Speech File")

        with gr.Row():
            result_box = gr.HTML(label="Prediction")

        with gr.Row():
            feature_plot_out = gr.Plot(label="Feature Heatmaps (WCQCC vs MFCC)")

        audio_input.change(
            fn=gradio_predict,
            inputs=audio_input,
            outputs=[result_box, feature_plot_out],
        )

    demo.launch(share=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to trained RandomForest .pkl file")
    parser.add_argument("--file",  default=None,
                        help="Single file to predict (skips Gradio UI)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Train first: python scripts/train_eval.py --data data/LA --feature wcqcc")
        return

    model = joblib.load(args.model)

    if args.file:
        label, conf, explanation = predict(args.file, model)
        print(f"\nResult: {label} ({conf*100:.1f}% confidence)")
        print(explanation)
    else:
        launch_gradio(model)


if __name__ == "__main__":
    main()
