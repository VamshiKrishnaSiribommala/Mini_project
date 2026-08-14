"""
LungXpert - Command Line Interface (CLI) Manual Analysis Tool
Usage:
    python run_cli_analysis.py
    python run_cli_analysis.py --dcm demo\sample_test.dcm
    python run_cli_analysis.py --dcm data\openi\dicom\1202_IM-0136-1001.dcm --text "Lobar consolidation in right lower lobe"
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src to python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer

from Helpers import Config, compare_maps
from DataHandler import DICOMImagePreprocessor, parse_openi_xml
from Model import MultiModalRetrievalModel
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
from Retrieval import make_retrieval_engine

def parse_args():
    parser = argparse.ArgumentParser(description="LungXpert CLI Analysis Tool")
    parser.add_argument("--dcm", "--dicom", type=str, default="", help="Path to DICOM image file (.dcm)")
    parser.add_argument("--text", type=str, default="", help="Clinical findings/radiology report text")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold for positive findings (default: 0.50)")
    parser.add_argument("--topk", type=int, default=5, help="Number of top retrieved reference cases (default: 5)")
    parser.add_argument("--out_dir", type=str, default="outputs/cli_manual_results", help="Directory to save generated explainability heatmaps")
    return parser.parse_args()

def save_overlay(orig_arr, heatmap, out_path, title="Heatmap Overlay", cmap="jet", alpha=0.45):
    """Saves a high-resolution PNG image overlay."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Resize heatmap to match image if necessary
    h, w = orig_arr.shape[:2]
    if heatmap is not None and (heatmap.shape[0] != h or heatmap.shape[1] != w):
        t_hm = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t_hm = F.interpolate(t_hm, size=(h, w), mode="bilinear", align_corners=False)
        heatmap = t_hm.squeeze().numpy()
        
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(orig_arr, cmap="gray")
    if heatmap is not None:
        norm_hm = (heatmap - np.nanmin(heatmap)) / (np.nanmax(heatmap) - np.nanmin(heatmap) + 1e-8)
        ax.imshow(norm_hm, cmap=cmap, alpha=alpha)
    ax.set_title(title, fontsize=12, pad=10, weight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    args = parse_args()

    print("\n" + "="*70)
    print("      LUNGXPERT: MULTIMODAL THORACIC ABNORMALITY ANALYSIS (CLI)")
    print("="*70)

    # Interactive input if not provided via flags and running in an interactive terminal
    dcm_path_str = args.dcm.strip()
    if not dcm_path_str:
        if sys.stdin and sys.stdin.isatty():
            print("\n[PROMPT] Enter path to DICOM file (.dcm)")
            print("         Press ENTER to use default: demo/sample_test.dcm")
            try:
                user_in = input("DICOM Path > ").strip().strip('"').strip("'")
            except EOFError:
                user_in = ""
            dcm_path_str = user_in if user_in else "demo/sample_test.dcm"
        else:
            dcm_path_str = "demo/sample_test.dcm"

    dcm_path = Path(dcm_path_str)
    if not dcm_path.is_absolute():
        dcm_path = BASE_DIR / dcm_path

    if not dcm_path.exists():
        print(f"\n[ERROR] File not found: {dcm_path}")
        sys.exit(1)

    text_query = args.text.strip()
    if not text_query:
        if sys.stdin and sys.stdin.isatty():
            print("\n[PROMPT] Enter clinical radiology narrative / findings")
            print("         Press ENTER for default: 'Normal chest x-ray with clear lungs.'")
            try:
                user_in = input("Report Text > ").strip()
            except EOFError:
                user_in = ""
            text_query = user_in if user_in else "Normal chest x-ray with clear lungs. Heart size is normal. No focal consolidation, pneumothorax, or pleural effusion."
        else:
            text_query = "Normal chest x-ray with clear lungs. Heart size is normal. No focal consolidation, pneumothorax, or pleural effusion."

    print("\n" + "-"*70)
    print(f"  • Selected Radiograph: {dcm_path.name} ({dcm_path.stat().st_size:,} bytes)")
    print(f"  • Full File Path:      {dcm_path}")
    print(f"  • Clinical Narrative:  \"{text_query}\"")
    print(f"  • Output Folder:       {args.out_dir}")
    print("-"*70)

    # Device & config setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/5] Initializing engine on {device.type.upper()}...")
    cfg = Config.load(BASE_DIR / "configs" / "config.yaml")

    # Labels
    label_cols = (
        list(disease_groups.keys()) +
        list(normal_groups.keys()) +
        list(finding_groups.keys()) +
        list(symptom_groups.keys())
    )
    labels_df = pd.read_csv(BASE_DIR / "outputs" / "openi_labels_final.csv").set_index("id")

    # Load Model
    print("[2/5] Loading trained multimodal model weights (checkpoints/model_best.pt)...")
    ckpt_path = BASE_DIR / "checkpoints" / "model_best.pt"
    model = MultiModalRetrievalModel(
        img_backbone="swin",
        num_fusion_layers=cfg.num_fusion_layers,
        joint_dim=cfg.joint_dim,
        num_heads=cfg.num_heads,
        fusion_type=cfg.fusion_type,
        swin_ckpt_path=BASE_DIR / "models" / "swin_checkpoint.safetensors",
        bert_local_dir=BASE_DIR / "models" / "clinicalbert_local",
        use_shared_ffn=cfg.use_shared_ffn,
        use_cls_only=cfg.use_cls_only,
        num_classes=len(label_cols),
        device=device,
        checkpoint_path=str(ckpt_path) if ckpt_path.exists() else None
    ).to(device)
    model.eval()

    # Preprocessor & Tokenizer
    print("[3/5] Initializing preprocessors and ClinicalBERT tokenizer...")
    preproc = DICOMImagePreprocessor()
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_DIR / "models" / "clinicalbert_local"),
        local_files_only=True
    )

    # Preprocess Inputs
    print("[4/5] Preprocessing DICOM radiograph and tokenizing clinical text...")
    with open(dcm_path, "rb") as f:
        dcm_bytes = f.read()
    raw_tensor = preproc(dcm_bytes)
    img_tensor = raw_tensor.unsqueeze(0).to(device)

    tokens = tokenizer(
        text_query,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=getattr(cfg, "text_dim", 128)
    )
    txt_ids = tokens.input_ids.to(device)
    txt_mask = tokens.attention_mask.to(device)

    # Convert display image
    orig_arr = raw_tensor.squeeze().cpu().numpy()
    if orig_arr.ndim == 3 and orig_arr.shape[0] in (1, 3):
        orig_arr = np.transpose(orig_arr, (1, 2, 0))
        if orig_arr.shape[2] == 1:
            orig_arr = orig_arr[:, :, 0]
    minv, maxv = float(np.nanmin(orig_arr)), float(np.nanmax(orig_arr))
    if maxv - minv > 1e-6:
        orig_arr = (orig_arr - minv) / (maxv - minv)
    else:
        orig_arr = np.clip(orig_arr, 0.0, 1.0)

    # Run Prediction + Explanation
    print("[5/5] Executing multimodal inference, Grad-CAM, and case retrieval...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.predict(img_tensor, txt_ids, txt_mask, K=args.topk, explain=True)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Extract Probabilities
    probs = out.get("probs", None)
    if probs is not None:
        if isinstance(probs, torch.Tensor):
            probs = probs[0].detach().cpu().numpy()
        elif isinstance(probs, np.ndarray):
            probs = probs[0] if probs.ndim > 1 else probs
    else:
        logits = out.get("logits", None)
        if isinstance(logits, torch.Tensor):
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        else:
            probs = logits[0] if logits.ndim > 1 else logits

    # Active predictions
    active_findings = []
    for idx, p in enumerate(probs):
        if p >= args.threshold:
            active_findings.append((label_cols[idx], float(p)))
    active_findings.sort(key=lambda x: x[1], reverse=True)

    # Top 5 Ranked Probabilities
    ranked_indices = np.argsort(probs)[::-1][:5]

    # Save Output Visualizations
    out_dir = BASE_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Original X-Ray
    path_orig = out_dir / "1_original_xray.png"
    save_overlay(orig_arr, None, path_orig, title="Original Radiograph (DICOM Windowed)")

    # 2. Grad-CAM
    gradcam_maps = out.get("gradcam_maps", {}) or {}
    path_gc = out_dir / "2_gradcam_overlay.png"
    target_idx = ranked_indices[0]
    gc_map = None
    if target_idx in gradcam_maps:
        gc_map = gradcam_maps[target_idx]
        if torch.is_tensor(gc_map):
            gc_map = gc_map.detach().cpu().numpy()
        save_overlay(orig_arr, gc_map, path_gc, title=f"Grad-CAM Heatmap ({label_cols[target_idx]})", cmap="jet")

    # 3. Integrated Gradients
    ig_maps = out.get("ig_maps", {}) or {}
    path_ig = out_dir / "3_integrated_gradients.png"
    ig_map = None
    if target_idx in ig_maps:
        ig_map = ig_maps[target_idx]
        if torch.is_tensor(ig_map):
            ig_map = ig_map.detach().cpu().numpy()
        save_overlay(orig_arr, ig_map, path_ig, title=f"Integrated Gradients ({label_cols[target_idx]})", cmap="viridis")

    # 4. Cross-Attention Text-to-Image Rollout
    att_maps = out.get("attention_map", {}) or {}
    path_attn = out_dir / "4_cross_attention_txt2img.png"
    if "txt2img" in att_maps and att_maps["txt2img"] is not None:
        t2i = att_maps["txt2img"]
        if torch.is_tensor(t2i):
            t2i = t2i.detach().cpu().numpy()
        save_overlay(orig_arr, t2i, path_attn, title="Cross-Modal Attention Rollout (Text -> Image)", cmap="plasma")

    # 5. Combined Salience Map
    path_comb = out_dir / "5_combined_salience.png"
    if "comb_img" in att_maps and att_maps["comb_img"] is not None:
        ci = att_maps["comb_img"]
        if torch.is_tensor(ci):
            ci = ci.detach().cpu().numpy()
        save_overlay(orig_arr, ci, path_comb, title="Harmonized Multimodal Salience Map", cmap="inferno")

    # =========================================================================
    # DISPLAY RESULTS IN TERMINAL
    # =========================================================================
    print("\n" + "="*70)
    print("                     MANUAL ANALYSIS RESULTS")
    print("="*70)
    print(f"Total Pipeline Latency: {latency_ms:.1f} ms")

    print("\n[A] THRESHOLD-BASED ACTIVE FINDINGS (Threshold >= {:.2f}):".format(args.threshold))
    if active_findings:
        for name, p in active_findings:
            bar = "#" * int(p * 30)
            print(f"   [+] {name:<28} : {p*100:6.2f}%  [{bar:<30}]")
    else:
        print("   (No finding exceeded threshold)")

    print("\n[B] TOP 5 PREDICTED ABNORMALITY PROBABILITIES:")
    for idx in ranked_indices:
        name = label_cols[idx]
        p = float(probs[idx])
        bar = "=" * int(p * 30)
        print(f"   • {name:<30} : {p*100:6.2f}%  [{bar:<30}]")

    print("\n[C] GENERATED VISUAL EXPLANATION ARTIFACTS (SAVED TO DISK):")
    print(f"   1. Original X-Ray Image   : file:///{path_orig.as_posix()}")
    if path_gc.exists():
        print(f"   2. Grad-CAM Localization  : file:///{path_gc.as_posix()}")
    if path_ig.exists():
        print(f"   3. Integrated Gradients   : file:///{path_ig.as_posix()}")
    if path_attn.exists():
        print(f"   4. Cross-Attention Rollout: file:///{path_attn.as_posix()}")
    if path_comb.exists():
        print(f"   5. Combined Salience Map  : file:///{path_comb.as_posix()}")

    # Retrieval Engine Query
    print("\n[D] CASE-BASED REFERENCE RETRIEVAL (PACS GALLERY SEARCH):")
    try:
        retriever = make_retrieval_engine(
            kind="dls",
            features_path=str(BASE_DIR / "embeddings" / "train_joint_embeddings.npy"),
            ids_path=str(BASE_DIR / "embeddings" / "train_ids.json"),
            fdb_path=str(BASE_DIR / "featureDBs" / "train_joint_embeddings_link_graph.pkl")
        )
        # Extract query joint vector
        joint_emb = out.get("joint_emb", None)
        if joint_emb is not None:
            q_vec = joint_emb.detach().cpu().numpy()
            ret_ids, ret_scores = retriever.retrieve(q_vec, K=args.topk)
            for rank, (rid, sc) in enumerate(zip(ret_ids, ret_scores), start=1):
                labels_str = "N/A"
                if rid in labels_df.index:
                    row = labels_df.loc[rid]
                    active_l = [col for col in label_cols if row.get(col, 0) == 1]
                    labels_str = ", ".join(active_l[:3]) if active_l else "Normal"
                print(f"   Rank #{rank}: Case ID: {rid:<22} | Score: {sc*100:5.1f}% | Diagnoses: {labels_str}")
    except Exception as e:
        print(f"   [INFO] PACS Retrieval query note: {e}")

    print("\n" + "="*70)
    print(f"[SUCCESS] All output images saved in: {out_dir}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
