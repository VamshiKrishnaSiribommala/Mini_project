#!/usr/bin/env python3
"""
LungXpert Checkpoint & Weight Downloader
Downloads large pre-trained model weights (Swin backbone and full multimodal model_best.pt)
from official Hugging Face repositories if they are not already present locally.
"""

import sys
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = BASE_DIR / "checkpoints"
MODEL_DIR = BASE_DIR / "models"

def download_multimodal_checkpoint():
    """Download the 1.52 GB trained multimodal checkpoint from Hugging Face if missing."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    target_path = CKPT_DIR / "model_best.pt"
    
    if target_path.exists() and target_path.stat().st_size > 100 * 1024 * 1024:
        print(f"[OK] Checkpoint already present at {target_path} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
        return target_path
    
    print(f"[INFO] Downloading model_best.pt from Hugging Face hub (repo: ppddddpp/unified-multimodal-chestxray)...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="ppddddpp/unified-multimodal-chestxray",
            filename="model/checkpoints/swin_i/model_best.pt"
        )
        shutil.copy2(downloaded, str(target_path))
        print(f"[SUCCESS] Downloaded and saved checkpoint to {target_path}")
        return target_path
    except Exception as e:
        print(f"[ERROR] Failed to download checkpoint: {e}", file=sys.stderr)
        print("[HINT] Ensure network connectivity and run: pip install huggingface-hub", file=sys.stderr)
        return None

def download_swin_backbone():
    """Download or verify the Swin-B vision backbone checkpoint."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target_path = MODEL_DIR / "swin_checkpoint.safetensors"
    
    if target_path.exists() and target_path.stat().st_size > 10 * 1024 * 1024:
        print(f"[OK] Swin backbone already present at {target_path} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
        return target_path
        
    print(f"[INFO] Downloading and adapting Swin Transformer backbone...")
    try:
        from src.Helpers.swinDownload import download_swin
        download_swin(swin_name='swin_base_patch4_window7_224', swin_ckpt_path=target_path, swin_channels=3)
        print(f"[SUCCESS] Swin backbone saved to {target_path}")
        return target_path
    except Exception as e:
        print(f"[ERROR] Failed to download Swin backbone: {e}", file=sys.stderr)
        return None

def main():
    print("=" * 65)
    print("LungXpert Automated Asset & Checkpoint Verification")
    print("=" * 65)
    
    ckpt = download_multimodal_checkpoint()
    swin = download_swin_backbone()
    
    print("\nSummary:")
    print(f"  • Multimodal Checkpoint: {'READY' if ckpt else 'FAILED'}")
    print(f"  • Swin Backbone:         {'READY' if swin else 'FAILED'}")
    print("=" * 65)

if __name__ == "__main__":
    main()
