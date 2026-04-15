import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
import os

# ─── 1. OPTIMIZED SOTA HYPER-PARAMETERS ──────────────────────────────────────
EMBED_DIM = 768
BATCH_SIZE = 48
LR_BACKBONE = 2e-5   # Slightly increased to recover from 72%
LR_NOVELTY = 2e-4    # Increased to re-sync Fourier & Slots
EPOCHS = 50          # More epochs to ensure we cross 96%
NUM_SLOTS = 8

# ─── 2. AUDIO LOADING (STRICT NORMALIZATION) ─────────────────────────────────
def audio_loader(path):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=128, n_fft=400, hop_length=160
    )
    mel = torch.log(mel_transform(wav) + 1e-9)
    
    if mel.shape[2] > 512: mel = mel[:, :, :512]
    elif mel.shape[2] < 512: mel = torch.nn.functional.pad(mel, (0, 512 - mel.shape[2]))
    
    # Global normalization constants from SSLAM Paper
    mel = (mel - (-4.268)) / (4.569 * 2)
    return mel.squeeze(0).transpose(0, 1)

class ESC50Dataset(Dataset):
    def __init__(self, audio_dir, loader):
        self.audio_dir = audio_dir
        self.loader = loader
        self.file_list = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        target = int(file_name.split('-')[-1].split('.')[0])
        return self.loader(os.path.join(self.audio_dir, file_name)), target

# ─── 3. MODEL ARCHITECTURE (FOURIER + SLOT ATTENTION) ───────────────────────
class MultiSlotPooling(nn.Module):
    def __init__(self, num_slots=8, d_model=768):
        super().__init__()
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.randn(1, num_slots, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
    def forward(self, x):
        b = x.shape[0]
        slots = self.slots.expand(b, -1, -1)
        slots_out, _ = self.attn(slots, x, x)
        return slots_out.mean(dim=1)

class SSLAM_SOTA_Final(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        fft_feature_dim = patch_size * (patch_size // 2 + 1)
        self.fourier_proj = nn.Linear(fft_feature_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, batch_first=True)
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.slot_pooling = MultiSlotPooling(num_slots=NUM_SLOTS, d_model=embed_dim)
        self.head = nn.Linear(embed_dim, 50)

    def forward(self, x):
        B, T, F = x.shape
        p = self.patch_size
        x = x.unfold(1, p, p).unfold(2, p, p)
        x = x.contiguous().view(B, -1, p, p)
        x_fft = torch.fft.rfft2(x, norm="ortho")
        log_mag = torch.log1p(torch.abs(x_fft)).view(B, x.shape[1], -1) 
        x = self.fourier_proj(log_mag)
        x = self.backbone(x)
        x = self.slot_pooling(x)
        return self.head(x)

# ─── 4. TRAINING WITH WARM RESTARTS ──────────────────────────────────────────
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLAM_SOTA_Final().to(device)
    DATA_PATH = "/teamspace/studios/this_studio/SSLAM2PRO/SSLAM/ESC-50/ESC-50_dataset"

    # CRITICAL: Always try to load the BEST checkpoint first
    checkpoint_path = "checkpoint_best.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {checkpoint_path} for recovery.")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LR_BACKBONE},
        {'params': model.fourier_proj.parameters(), 'lr': LR_NOVELTY},
        {'params': model.slot_pooling.parameters(), 'lr': LR_NOVELTY},
        {'params': model.head.parameters(), 'lr': LR_NOVELTY}
    ], weight_decay=0.05)
    
    # Prevents getting stuck in local minima
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    train_loader = DataLoader(ESC50Dataset(DATA_PATH, audio_loader), batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n--- RECOVERY MISSION: TARGET 96.2%+ ---")
    best_acc = 0.0 # Reset to track new best in this run
    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        scheduler.step()
        acc = 100.*correct/total
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc,
            }, "checkpoint_SOTA_96.pt")
            print(f" ✓ Progress: {acc:.2f}% - Saved as checkpoint_SOTA_96.pt")
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Acc: {acc:.2f}% | Best this run: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()