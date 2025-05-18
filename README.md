# Binary Classification with PyTorch Models

This project provides a binary image classification training pipeline using PyTorch. You can choose among three different models to train your classifier:

- **ReXNet v1.0** (`rex`)
- **ResNet-18** (`res`)
- **Custom Model** (`cs`)

---

## 🔧 Requirements

Install the dependencies using pip:

```bash
pip install torch torchvision torchaudio
```

---

## 🚀 Run Training

You can start training with one of the following commands:

### ✅ Use ReXNet v1.0

```bash
python train.py --model rex
```

### ✅ Use ResNet-18

```bash
python train.py --model res
```

### ✅ Use Custom Model

```bash
python train.py --model cs
```

---

## ⚙️ Example Training Parameters

You can use model parameter to decide which model you want to use:

```bash
python train.py --model rex
```

---

## 📦 Model Summary

| Alias | Model         | Notes                            |
|-------|---------------|----------------------------------|
| rex   | ReXNet v1.0   | Mobile-friendly, efficient       |
| res   | ResNet-18     | Lightweight CNN                  |
| cs    | Custom Model  | model from `model.py`  |

---

## 📁 Files

- `train.py` — main training script
- `model.py` — custom model (`cs`)
- `rexnet_v1.py` ReXNet v1.

---

## 🙋 FAQ

---

## 📜 License

MIT License