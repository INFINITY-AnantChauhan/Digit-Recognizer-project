import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "model/digit_model.h5"
CM_PATH    = "model/confusion_matrix.png"
N_EXAMPLES = 10     # number of misclassified examples to display


# ── 1. Load model & data ──────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading MNIST test set...")
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0

print(f"    Test samples : {len(x_test)}")


# ── 2. Predict ────────────────────────────────────────────────────────────────
print("\nRunning inference...")
probs  = model.predict(x_test, verbose=0)          # shape: (10000, 10)
y_pred = np.argmax(probs, axis=1)                  # most likely class
y_conf = np.max(probs, axis=1)                     # confidence of prediction


# ── 3. Overall metrics ────────────────────────────────────────────────────────
test_loss = tf.keras.losses.sparse_categorical_crossentropy(y_test, probs)
test_loss = float(tf.reduce_mean(test_loss).numpy())
test_acc  = float(np.mean(y_pred == y_test))

print(f"\n{'='*45}")
print(f"  Overall Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Overall Test Loss     : {test_loss:.4f}")
print(f"  Total Errors          : {int((y_pred != y_test).sum())} / {len(y_test)}")
print(f"  Mean Confidence       : {y_conf.mean():.4f}")
print(f"{'='*45}\n")


# ── 4. Per-class report ───────────────────────────────────────────────────────
print("Per-class Classification Report:")
print("-" * 55)
print(classification_report(
    y_test, y_pred,
    target_names=[f"Digit {i}" for i in range(10)]
))


# ── 5. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(9, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=list(range(10)))
disp.plot(cmap="Blues", ax=ax, colorbar=True)
ax.set_title("MNIST Test Set — Confusion Matrix", fontsize=14, pad=12)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
plt.tight_layout()
plt.savefig(CM_PATH, dpi=150, bbox_inches="tight")
print(f"Confusion matrix saved → {CM_PATH}")
plt.show()


# ── 6. Most confused digit pairs ─────────────────────────────────────────────
print("\nTop 10 most confused pairs (true → predicted):")
print(f"  {'True':>6}  {'Predicted':>10}  {'Count':>6}")
print("  " + "-" * 28)

off_diag = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            off_diag.append((i, j, cm[i, j]))
off_diag.sort(key=lambda x: -x[2])

for true, pred, count in off_diag[:10]:
    print(f"  {true:>6}  {pred:>10}  {count:>6}")


# ── 7. Show misclassified examples ────────────────────────────────────────────
error_idx = np.where(y_pred != y_test)[0]
sample_idx = np.random.choice(error_idx, size=min(N_EXAMPLES, len(error_idx)),
                               replace=False)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()
for ax, idx in zip(axes, sample_idx):
    ax.imshow(x_test[idx], cmap="gray")
    ax.set_title(
        f"True: {y_test[idx]}  Pred: {y_pred[idx]}\n"
        f"Conf: {y_conf[idx]*100:.1f}%",
        fontsize=9, color="red"
    )
    ax.axis("off")
plt.suptitle("Misclassified Examples", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("model/misclassified_examples.png", dpi=150, bbox_inches="tight")
print("Misclassified examples saved → model/misclassified_examples.png")
plt.show()


# ── 8. Per-digit accuracy bar chart ──────────────────────────────────────────
per_digit_acc = [
    np.mean(y_pred[y_test == d] == d) for d in range(10)
]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(range(10), [a * 100 for a in per_digit_acc],
              color="steelblue", edgecolor="white", linewidth=0.5)
ax.set_xticks(range(10))
ax.set_xlabel("Digit")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Per-digit Accuracy on Test Set")
ax.set_ylim(95, 100.5)
ax.grid(axis="y", alpha=0.3)
for bar, acc in zip(bars, per_digit_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("model/per_digit_accuracy.png", dpi=150, bbox_inches="tight")
print("Per-digit accuracy chart saved → model/per_digit_accuracy.png")
plt.show()
