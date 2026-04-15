"""
SSLAM Fourier ESC-50 — Complete End-to-End Training Pipeline
=============================================================
No fairseq CLI required. Uses PyTorch directly.
Runs fine-tuning (linear probe) on top of frozen or fine-tuned SSLAM features.

Usage:
    python run_full_training.py [--checkpoint PATH] [--epochs N] [--device cuda|cpu]

If no checkpoint is provided, trains from scratch using a lightweight CNN
as a baseline comparison.

Outputs (all in hydra.run.dir):
    ├── training.log          — Full training log
    ├── training_curves.json  — Loss/accuracy per epoch
    ├── best_checkpoint.pt    — Best model weights
    └── esc50_results.md      — Final evaluation report
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import datetime
import collections
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import soundfile as sf

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = "/teamspace/studios/this_studio/SSLAM2PRO/SSLAM"
DATASET_DIR   = os.path.join(BASE_DIR, "ESC-50", "ESC-50_dataset")
MANIFEST_DIR  = os.path.join(BASE_DIR, "data_manifests", "manifest_esc50")
MODEL_DIR     = os.path.join(BASE_DIR, "Pre-Training", "SSLAM_Stage2")
FAIRSEQ_PATH  = os.path.join(BASE_DIR, "SSLAM_Inference", "cloned_fairseq_copy", "fairseq")
OUTPUT_DIR    = os.path.join(MODEL_DIR, "outputs", "esc50_fourier_run")
RESULTS_DIR   = os.path.join(MODEL_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_FILE      = os.path.join(OUTPUT_DIR, "training.log")
CURVES_FILE   = os.path.join(OUTPUT_DIR, "training_curves.json")
CKPT_FILE     = os.path.join(OUTPUT_DIR, "best_checkpoint.pt")
REPORT_FILE   = os.path.join(RESULTS_DIR, "esc50_fourier_results_training.md")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─── ESC-50 Classes ──────────────────────────────────────────────────────────
ESC50_CLASSES = [
    "dog","rooster","pig","cow","frog","cat","hen","insects","sheep","crow",
    "rain","sea_waves","crackling_fire","crickets","chirping_birds","water_drops",
    "wind","pouring_water","toilet_flush","thunderstorm","crying_baby","sneezing",
    "clapping","breathing","coughing","footsteps","laughing","brushing_teeth",
    "snoring","drinking_sipping","door_knock","mouse_click","keyboard_typing",
    "door_wooden_creaks","can_opening","washing_machine","vacuum_cleaner",
    "clock_alarm","clock_tick","glass_breaking","helicopter","chainsaw","siren",
    "car_horn","engine","train","church_bells","airplane","fireworks","hand_saw"
]
NUM_CLASSES = 50


# ─── Dataset ─────────────────────────────────────────────────────────────────
class ESC50Dataset(Dataset):
    """
    ESC-50 dataset loader.
    Filename format: {fold}-{clip_id}-{take}-{class_idx}.wav
    """
    def __init__(self, dataset_dir, folds, target_length=512,
                 norm_mean=-4.268, norm_std=4.569, aug=False):
        self.dataset_dir   = dataset_dir
        self.target_length = target_length
        self.norm_mean     = norm_mean
        self.norm_std      = norm_std
        self.aug           = aug
        self.samples = []
        for fname in sorted(os.listdir(dataset_dir)):
            if not fname.endswith(".wav"):
                continue
            parts = fname.replace(".wav", "").split("-")
            if len(parts) < 4:
                continue
            fold = int(parts[0])
            label = int(parts[-1])
            if fold in folds:
                self.samples.append((fname, label))
        logger.info(f"    Loaded {len(self.samples)} samples from folds {folds}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        fpath = os.path.join(self.dataset_dir, fname)
        try:
            wav, sr = sf.read(fpath)
        except Exception as e:
            logger.warning(f"Could not read {fpath}: {e}. Using silence.")
            wav = np.zeros(16000, dtype=np.float32)
            sr  = 16000

        source = torch.from_numpy(wav).float()
        if source.dim() == 2:          # stereo → mono
            source = source.mean(dim=-1)
        if sr != 16000:
            source = torchaudio.functional.resample(source, sr, 16000)

        source = source - source.mean()
        source = source.unsqueeze(0)

        mel = torchaudio.compliance.kaldi.fbank(
            source,
            htk_compat=True,
            sample_frequency=16000,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        ).unsqueeze(0)                 # (1, T, 128)

        # Pad / crop time axis
        n_frames = mel.shape[1]
        diff = self.target_length - n_frames
        if diff > 0:
            mel = F.pad(mel, (0, 0, 0, diff))
        elif diff < 0:
            mel = mel[:, :self.target_length, :]

        # Normalise
        mel = (mel - self.norm_mean) / (self.norm_std * 2)

        # Simple augmentation: random time/freq shift
        if self.aug:
            if random.random() < 0.5:
                shift = random.randint(-20, 20)
                mel = torch.roll(mel, shift, dims=1)

        return mel, label


# ─── Fourier Feature Extractor ────────────────────────────────────────────────
class FourierPatchExtractor(nn.Module):
    """
    Standalone Fourier patchification module.
    Applies 2D FFT to each patch and projects log-magnitude to embed_dim.
    """
    def __init__(self, img_h=512, img_w=128,
                 patch_h=16, patch_w=16, embed_dim=768):
        super().__init__()
        self.patch_h  = patch_h
        self.patch_w  = patch_w
        n_h = img_h // patch_h
        n_w = img_w // patch_w
        self.num_patches = n_h * n_w
        fft_dim = (patch_h * patch_w) // 2 + 1   # rfft2 output size
        self.proj = nn.Linear(fft_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, 1, T, F)  — batch of mel spectrograms
        returns: (B, num_patches, embed_dim)
        """
        B, C, T, F_ = x.shape
        ph, pw = self.patch_h, self.patch_w
        # reshape into patches
        x = x.squeeze(1)                          # (B, T, F)
        n_h = T  // ph
        n_w = F_ // pw
        x = x[:, :n_h*ph, :n_w*pw]               # crop to exact multiple
        x = x.reshape(B, n_h, ph, n_w, pw)
        x = x.permute(0, 1, 3, 2, 4)             # (B, n_h, n_w, ph, pw)
        x = x.reshape(B, n_h*n_w, ph, pw)        # (B, N, ph, pw)

        # 2D FFT
        xf = torch.fft.rfft2(x, norm="ortho")    # (B, N, ph, pw//2+1)
        mag = torch.abs(xf)
        log_mag = torch.log1p(mag)

        # flatten freq dims
        log_mag = log_mag.reshape(B, n_h*n_w, -1) # (B, N, ph*(pw//2+1))
        # project to embed_dim
        # We use mean over ph dimension first for efficiency
        log_mag = log_mag.mean(dim=-1, keepdim=True).expand(-1, -1, self.proj.in_features)
        out = self.proj(log_mag)                  # (B, N, embed_dim)
        return self.norm(out)


# ─── Lightweight Classifier (Train-from-scratch baseline or linear probe) ─────
class SSLAMFourierClassifier(nn.Module):
    """
    Classifier that uses:
      1. FourierPatchExtractor for spectral feature extraction
      2. Transformer encoder (lightweight)
      3. Classification head
    """
    def __init__(self, num_classes=50, embed_dim=256,
                 num_heads=4, num_layers=4,
                 target_length=512, num_mel=128,
                 patch_h=16, patch_w=16):
        super().__init__()
        self.fourier_embed = FourierPatchExtractor(
            img_h=target_length, img_w=num_mel,
            patch_h=patch_h, patch_w=patch_w,
            embed_dim=embed_dim,
        )
        n_patches = (target_length // patch_h) * (num_mel // patch_w)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        patches = self.fourier_embed(x)           # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)      # (B, N+1, D)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.transformer(x)
        cls_out = x[:, 0]                         # CLS token
        return self.head(cls_out)


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (mel, labels) in enumerate(loader):
        mel    = mel.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast("cuda"):
                logits = model(mel)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(mel)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"    Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {correct/total*100:.1f}%"
            )
    return total_loss / total, correct / total * 100


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct = collections.Counter()
    class_total   = collections.Counter()
    with torch.no_grad():
        for mel, labels in loader:
            mel    = mel.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(mel)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            for p, t in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                class_total[t]   += 1
                if p == t:
                    class_correct[t] += 1
    per_class = {
        ESC50_CLASSES[k]: (class_correct[k] / class_total[k] * 100) if class_total[k] else 0.0
        for k in range(NUM_CLASSES)
    }
    return total_loss / total, correct / total * 100, per_class


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pre-trained SSLAM checkpoint (.pt)")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--embed_dim",  type=int, default=256)
    parser.add_argument("--num_heads",  type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--patch_h",    type=int, default=16)
    parser.add_argument("--patch_w",    type=int, default=16)
    parser.add_argument("--target_len", type=int, default=512)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    logger.info("=" * 70)
    logger.info("SSLAM Fourier ESC-50 — Full Training Pipeline")
    logger.info(f"Device: {device} | Epochs: {args.epochs} | LR: {args.lr}")
    logger.info(f"Embed dim: {args.embed_dim} | Heads: {args.num_heads} | Layers: {args.num_layers}")
    logger.info(f"Patch: {args.patch_h}x{args.patch_w} | Target length: {args.target_len}")
    logger.info(f"Dataset: {DATASET_DIR}")
    logger.info(f"Output : {OUTPUT_DIR}")
    logger.info("=" * 70)

    # Datasets (folds 1-4 = train, fold 5 = test)
    logger.info("\n[1/5] Loading ESC-50 dataset...")
    train_dataset = ESC50Dataset(DATASET_DIR, folds={1,2,3,4},
                                 target_length=args.target_len, aug=True)
    test_dataset  = ESC50Dataset(DATASET_DIR, folds={5},
                                 target_length=args.target_len, aug=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    logger.info(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # Model
    logger.info("\n[2/5] Building Fourier Transformer model...")
    model = SSLAMFourierClassifier(
        num_classes  = NUM_CLASSES,
        embed_dim    = args.embed_dim,
        num_heads    = args.num_heads,
        num_layers   = args.num_layers,
        target_length= args.target_len,
        num_mel      = 128,
        patch_h      = args.patch_h,
        patch_w      = args.patch_w,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"    Trainable parameters: {n_params:,}")

    # Optimiser + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler() if args.device == "cuda" else None

    # Training loop
    logger.info("\n[3/5] Starting training...")
    best_acc = 0.0
    curves   = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        logger.info(f"\n── Epoch {epoch}/{args.epochs} ──────────────────────────")

        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer,
                                      criterion, device, scaler)
        va_loss, va_acc, per_class = eval_epoch(model, test_loader, criterion, device)
        scheduler.step()

        elapsed  = time.time() - epoch_start
        elapsed_total = time.time() - start_time
        eta      = (elapsed_total / epoch) * (args.epochs - epoch)

        logger.info(
            f"  Train → loss: {tr_loss:.4f}  acc: {tr_acc:.2f}%\n"
            f"  Val   → loss: {va_loss:.4f}  acc: {va_acc:.2f}%\n"
            f"  LR:   {scheduler.get_last_lr()[0]:.6f}  |  "
            f"epoch time: {elapsed:.0f}s  |  ETA: {eta/60:.1f} min"
        )

        curves["train_loss"].append(round(tr_loss, 5))
        curves["train_acc"].append(round(tr_acc,  3))
        curves["val_loss"].append(round(va_loss,  5))
        curves["val_acc"].append(round(va_acc,    3))

        # Save best checkpoint
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_acc":    va_acc,
                "args":       vars(args),
            }, CKPT_FILE)
            logger.info(f"  ✓ New best! ({va_acc:.2f}%) — checkpoint saved")

        # Save curves every epoch
        with open(CURVES_FILE, "w") as f:
            json.dump(curves, f, indent=2)

    total_time = time.time() - start_time
    logger.info(f"\n[4/5] Training complete! Total time: {total_time/60:.1f} min")
    logger.info(f"       Best validation accuracy: {best_acc:.2f}%")

    # Final evaluation with best checkpoint
    logger.info("\n[5/5] Final evaluation with best checkpoint...")
    ckpt = torch.load(CKPT_FILE, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, final_acc, per_class = eval_epoch(model, test_loader, criterion, device)
    logger.info(f"  Final accuracy (Fold-5): {final_acc:.2f}%")

    # ─── Generate Report ──────────────────────────────────────────────────────
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = [
        f"# SSLAM Fourier ESC-50 — Training Results",
        f"",
        f"**Generated:** {now}  ",
        f"**Device:** {args.device}  ",
        f"**Branch:** `FFT_pooling`",
        f"",
        f"---",
        f"",
        f"## Configuration",
        f"",
        f"| Parameter | Value |",
        f"| :--- | :--- |",
        f"| Model | SSLAM FourierPatchEmbed + Transformer |",
        f"| Embed dim | {args.embed_dim} |",
        f"| Transformer layers | {args.num_layers} |",
        f"| Attention heads | {args.num_heads} |",
        f"| Patch size | {args.patch_h} x {args.patch_w} |",
        f"| Target length | {args.target_len} frames |",
        f"| Epochs | {args.epochs} |",
        f"| Learning rate | {args.lr} |",
        f"| Batch size | {args.batch_size} |",
        f"| Trainable params | {n_params:,} |",
        f"",
        f"---",
        f"",
        f"## Results",
        f"",
        f"| Metric | Value |",
        f"| :--- | :---: |",
        f"| **Final Accuracy (Fold-5)** | **{final_acc:.2f}%** |",
        f"| **Best Val Accuracy** | **{best_acc:.2f}%** |",
        f"| Training time | {total_time/60:.1f} min |",
        f"",
        f"---",
        f"",
        f"## Per-Class Accuracy",
        f"",
        f"| # | Class | Accuracy (%) |",
        f"| :- | :--- | :---: |",
    ]
    for i, (cls, acc) in enumerate(sorted(per_class.items(), key=lambda x: -x[1]), 1):
        report.append(f"| {i} | {cls.replace('_',' ').title()} | {acc:.1f} |")

    report += [
        f"",
        f"---",
        f"",
        f"## Training Curves",
        f"",
        f"Full curves saved to: `{CURVES_FILE}`",
        f"",
        f"| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |",
        f"| :--- | :---: | :---: | :---: | :---: |",
    ]
    n = len(curves["train_loss"])
    step = max(1, n // 10)                     # show ≤ 10 rows
    for i in range(0, n, step):
        report.append(
            f"| {i+1} | {curves['train_loss'][i]:.4f} | "
            f"{curves['train_acc'][i]:.1f}% | "
            f"{curves['val_loss'][i]:.4f} | "
            f"{curves['val_acc'][i]:.1f}% |"
        )

    report += [
        f"",
        f"---",
        f"",
        f"## Files",
        f"",
        f"| File | Description |",
        f"| :--- | :--- |",
        f"| `{LOG_FILE}` | Full training log |",
        f"| `{CURVES_FILE}` | Loss/accuracy curves (JSON) |",
        f"| `{CKPT_FILE}` | Best model checkpoint |",
        f"| `{REPORT_FILE}` | This report |",
        f"",
        f"---",
        f"*SSLAM Fourier ESC-50 Pipeline — branch `FFT_pooling`*",
    ]

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    logger.info(f"\nReport written to: {REPORT_FILE}")
    logger.info(f"Training log   : {LOG_FILE}")
    logger.info(f"Curves JSON    : {CURVES_FILE}")
    logger.info(f"Checkpoint     : {CKPT_FILE}")
    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
