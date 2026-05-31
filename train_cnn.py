"""
train_cnn.py — MNIST Digit Recognizer (PyTorch CNN)
=====================================================
Trains a Convolutional Neural Network on MNIST.
Achieves ~99%+ test accuracy.

Usage:
    python train_cnn.py

Requirements:
    pip install torch torchvision matplotlib
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS     = 10
BATCH_SIZE = 64
LR         = 1e-3
VAL_SPLIT  = 0.1
MODEL_PATH = "model/digit_model_cnn.pt"
DATA_DIR   = "./data"

os.makedirs("model", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ── 1. Data ───────────────────────────────────────────────────────────────────
print("\n[1/4] Loading MNIST dataset...")

transform_train = transforms.Compose([
    transforms.RandomRotation(8),           # slight rotation for robustness
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # slight shift
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform_train)
test_ds    = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform_test)

n_val    = int(len(full_train) * VAL_SPLIT)
n_train  = len(full_train) - n_val
train_ds, val_ds = random_split(full_train, [n_train, n_val],
                                generator=torch.Generator().manual_seed(SEED))

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"    Train : {len(train_ds)} samples")
print(f"    Val   : {len(val_ds)} samples")
print(f"    Test  : {len(test_ds)} samples")


# ── 2. CNN Model ──────────────────────────────────────────────────────────────
class MNISTConvNet(nn.Module):
    """
    CNN Architecture:
    Input (1x28x28)
      → Conv1 (32 filters, 3x3) → BN → ReLU → MaxPool
      → Conv2 (64 filters, 3x3) → BN → ReLU → MaxPool
      → Conv3 (128 filters, 3x3) → BN → ReLU
      → Flatten → Dense(256) → Dropout → Dense(10)
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 1x28x28 → 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 32x28x28 → 32x14x14

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x14x14 → 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 64x14x14 → 64x7x7

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64x7x7 → 128x7x7
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 128x7x7 → 6272
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),            # 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


print("\n[2/4] Building CNN model...")
model     = MNISTConvNet().to(device)
loss_fn   = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    Trainable parameters: {total_params:,}")


# ── Helper functions ──────────────────────────────────────────────────────────
def train_epoch(model, loader):
    model.train()
    total_loss, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (pred.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += loss_fn(pred, y).item() * len(y)
        correct    += (pred.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# ── 3. Training Loop ──────────────────────────────────────────────────────────
print("\n[3/4] Training CNN...")
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  "
      f"{'Val Loss':>10}  {'Val Acc':>10}  {'Time':>6}")
print("-" * 62)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(model, train_dl)
    vl_loss, vl_acc = eval_epoch(model, val_dl)
    elapsed = time.time() - t0

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)

    scheduler.step(vl_loss)

    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        torch.save(model.state_dict(), MODEL_PATH)
        flag = " ← best"
    else:
        flag = ""

    print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc*100:>9.2f}%  "
          f"{vl_loss:>10.4f}  {vl_acc*100:>9.2f}%  {elapsed:>5.1f}s{flag}")


# ── 4. Final Evaluation ───────────────────────────────────────────────────────
print("\n[4/4] Loading best model and evaluating on test set...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
_, test_acc = eval_epoch(model, test_dl)
print(f"    Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"\nModel saved → {MODEL_PATH}")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot([a*100 for a in history["train_acc"]], label="Train")
axes[0].plot([a*100 for a in history["val_acc"]],   label="Val")
axes[0].set_title("Accuracy (%)")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history["train_loss"], label="Train")
axes[1].plot(history["val_loss"],   label="Val")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("model/training_curves_cnn.png", dpi=150, bbox_inches="tight")
print("Training curves saved → model/training_curves_cnn.png")
plt.show()
