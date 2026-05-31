"""
predict.py — Run Inference using CNN Model (PyTorch)
=====================================================
Usage:
    python predict.py --test --n 10
    python predict.py --image my_digit.png
    python predict.py --interactive

Requirements:
    pip install torch torchvision numpy matplotlib pillow
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "model/digit_model_cnn.pt"
IMG_SIZE   = (28, 28)
device     = "cuda" if torch.cuda.is_available() else "cpu"


# ── CNN Model (must match train_cnn.py) ───────────────────────────────────────
class MNISTConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(path=MODEL_PATH):
    print(f"Loading CNN model from {path} ...")
    try:
        model = MNISTConvNet().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print("    Model loaded successfully.")
        return model
    except Exception as e:
        print(f"    ERROR: {e}")
        print("    Run train_cnn.py first!")
        sys.exit(1)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(path):
    img = Image.open(path).convert("L")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0

    # Invert if dark digit on white background
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    display = arr.copy()
    arr = (arr - 0.1307) / 0.3081
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,28,28)
    return tensor, display


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(model, tensor):
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    digit      = int(np.argmax(probs))
    confidence = float(probs[digit])
    return digit, confidence, probs


# ── Visualization ─────────────────────────────────────────────────────────────
def show_prediction(x_display, digit, confidence, probs,
                    true_label=None, save_path=None):
    fig = plt.figure(figsize=(10, 4))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.35)

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(x_display, cmap="gray", interpolation="nearest")
    ax_img.axis("off")
    title = f"Predicted: {digit}  ({confidence*100:.1f}%)"
    color = "black"
    if true_label is not None:
        color  = "green" if digit == true_label else "red"
        title += f"\nTrue: {true_label}"
    ax_img.set_title(title, fontsize=12, color=color, pad=8)

    ax_bar = fig.add_subplot(gs[1])
    colors = ["steelblue"] * 10
    colors[digit] = "limegreen" if (true_label is None or digit == true_label) else "tomato"
    bars = ax_bar.barh(range(10), probs * 100,
                       color=colors, edgecolor="white", linewidth=0.4)
    ax_bar.set_yticks(range(10))
    ax_bar.set_yticklabels([str(i) for i in range(10)], fontsize=10)
    ax_bar.set_xlabel("Probability (%)")
    ax_bar.set_title("Class Probabilities", fontsize=11)
    ax_bar.set_xlim(0, 110)
    ax_bar.invert_yaxis()
    ax_bar.grid(axis="x", alpha=0.3)
    for bar, p in zip(bars, probs):
        if p > 0.01:
            ax_bar.text(p * 100 + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{p*100:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


# ── Modes ─────────────────────────────────────────────────────────────────────
def predict_from_file(model, image_path):
    print(f"\nProcessing: {image_path}")
    try:
        tensor, display = preprocess_image(image_path)
    except FileNotFoundError:
        print(f"  ERROR: File not found — '{image_path}'")
        sys.exit(1)

    digit, confidence, probs = predict(model, tensor)

    print(f"\n  {'Predicted digit':<18}: {digit}")
    print(f"  {'Confidence':<18}: {confidence*100:.2f}%")
    print(f"\n  All probabilities:")
    for i, p in enumerate(probs):
        bar    = "█" * int(p * 40)
        marker = " ← predicted" if i == digit else ""
        print(f"    {i}: {bar:<42} {p*100:5.2f}%{marker}")

    save_path = image_path.rsplit(".", 1)[0] + "_prediction.png"
    show_prediction(display, digit, confidence, probs, save_path=save_path)


def predict_from_test_set(model, n=10):
    print(f"\nPredicting {n} random MNIST test samples...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    indices = np.random.choice(len(test_ds), size=n, replace=False)

    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 2, rows * 4))
    axes = np.array(axes).reshape(rows * 2, cols)

    correct = 0
    for k, idx in enumerate(indices):
        row_img = (k // cols) * 2
        row_bar = row_img + 1
        col     = k % cols

        tensor, true_label = test_ds[idx]
        true_label = int(true_label)
        tensor_in  = tensor.unsqueeze(0).to(device)
        digit, confidence, probs = predict(model, tensor_in)
        correct += (digit == true_label)

        raw = test_ds.data[idx].numpy().astype("float32") / 255.0

        ax_img = axes[row_img][col]
        ax_img.imshow(raw, cmap="gray")
        ax_img.axis("off")
        color = "green" if digit == true_label else "red"
        ax_img.set_title(f"P:{digit} T:{true_label}\n{confidence*100:.0f}%",
                         fontsize=8, color=color)

        ax_bar = axes[row_bar][col]
        colors = ["steelblue"] * 10
        colors[digit] = "green" if digit == true_label else "red"
        ax_bar.barh(range(10), probs, color=colors, height=0.7)
        ax_bar.set_yticks(range(10))
        ax_bar.set_yticklabels([str(i) for i in range(10)], fontsize=6)
        ax_bar.invert_yaxis()
        ax_bar.tick_params(axis="x", labelsize=6)
        ax_bar.set_xlim(0, 1.1)

    plt.suptitle(f"CNN Predictions — {correct}/{n} correct ({correct/n*100:.0f}%)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("model/test_predictions_cnn.png", dpi=150, bbox_inches="tight")
    print(f"\nAccuracy on sample: {correct}/{n} ({correct/n*100:.1f}%)")
    print("Grid saved → model/test_predictions_cnn.png")
    plt.show()


def interactive_mode(model):
    print("\nInteractive prediction mode (type 'quit' to exit)\n")
    while True:
        path = input("Enter image path: ").strip()
        if path.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        predict_from_file(model, path)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",       type=str)
    group.add_argument("--test",        action="store_true")
    group.add_argument("--interactive", action="store_true")
    parser.add_argument("--n",     type=int, default=10)
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    model = load_model(args.model)

    if args.image:
        predict_from_file(model, args.image)
    elif args.test:
        predict_from_test_set(model, n=args.n)
    elif args.interactive:
        interactive_mode(model)
