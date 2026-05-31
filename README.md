# 🔢 Handwritten Digit Recognizer — MNIST

A deep learning project that classifies handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Achieves **99.27% test accuracy**.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![Accuracy](https://img.shields.io/badge/Accuracy-99.27%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Results

| Training Curves |
<img width="1200" height="400" alt="Training Curves" src="https://github.com/user-attachments/assets/9c58d72d-3f9d-4b1d-85eb-d1f1693f4444" />
| Confusion Matrix |
<img width="900" height="684" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/ed070098-6841-4430-9ee7-0eee4ba5f870" />


| Per-digit Accuracy | 
<img width="900" height="400" alt="Pre-digit Accuracy on Test set" src="https://github.com/user-attachments/assets/3b8784a4-5409-4d95-926c-cf101f1ea0d5" />

| Sample Predictions |
<img width="1536" height="754" alt="Test Cases" src="https://github.com/user-attachments/assets/9c86723b-d5a9-47e6-875a-9a82a1a11ffd" />

---

## 🧠 Model Architecture

```
Input (1×28×28)
  │
  ├── Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  ├── Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  ├── Conv2D(128, 3×3) → BatchNorm → ReLU
  │
  └── Flatten → Dense(256) → Dropout(0.5) → Dense(10) → Softmax
```

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv Block 1 | 32×14×14 | 320 |
| Conv Block 2 | 64×7×7 | 18,496 |
| Conv Block 3 | 128×7×7 | 73,856 |
| Dense 1 | 256 | 1,606,912 |
| Output | 10 | 2,570 |
| **Total** | | **~1.7M** |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **99.27%** |
| Test Loss | 0.0309 |
| Training Epochs | 10 |
| Training Time | ~6 min (CPU) |

### Training Progress

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 92.13% | 97.55% |
| 5 | 97.89% | 98.28% |
| 10 | 98.56% | 99.03% |
| **Final Test** | — | **99.27%** |

---

## 📁 Project Structure

```
mnist-digit-recognizer/
├── train_cnn.py                      # CNN training script
├── predict.py                        # Inference on images or test set
├── evaluate.py                       # Confusion matrix & metrics
├── requirements.txt                  # Dependencies
├── Training Curves.png               # Accuracy & loss plots
├── Confusion Matrix.png              # Evaluation confusion matrix
├── Pre-digit Accuracy on Test set.png  # Per-digit accuracy chart
├── Test Cases.png                    # Sample prediction grid
├── digit.png                         # Sample input digit image
├── digit_prediction.png              # Prediction output for digit.png
├── data/                             # MNIST dataset (auto-downloaded)
└── model/
    └── digit_model_cnn.pt            # Trained model weights
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/mnist-digit-recognizer.git
cd mnist-digit-recognizer
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train_cnn.py
```

### 5. Evaluate
```bash
python evaluate.py
```

### 6. Predict
```bash
# On 10 random MNIST test samples
python predict.py --test --n 10

# On your own image
python predict.py --image digit.png

# Interactive mode
python predict.py --interactive
```

---

## 📦 Dataset

MNIST is auto-downloaded when you run `train_cnn.py`.

| Split | Samples |
|-------|---------|
| Train | 54,000 |
| Validation | 6,000 |
| Test | 10,000 |

Manual download: [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/)

---

## 🛠️ Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.26.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pillow>=10.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📈 Future Improvements

- [ ] Deploy as a web app using Streamlit
- [ ] Add a live drawing canvas (Tkinter)
- [ ] Try ResNet or deeper architectures
- [ ] Test on Fashion-MNIST dataset
- [ ] Export to ONNX for cross-platform inference

---

## 👤 Author

**Anant Chauhan**
- GitHub: [@INFINITY-AnantChauhan](https://github.com/INFINITY-AnantChauhan)

---

## 📄 License

This project is licensed under the MIT License.
