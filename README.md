# 🥔🍅🥕 NinjaCart Multiclass Vegetable Image Classification

## 📌 Problem Statement
Build an AI system to automatically classify images of **onion, potato, tomato** and filter out unrelated **Indian market** scenes (noise).  
Key challenge: robustly handle **varying lighting, angles, image sizes, and cluttered backgrounds** so the system can reduce manual sorting, decrease waste, and improve supply‑chain efficiency.

---

## 📂 Dataset
- **Classes:** `indian market` (noise), `onion`, `potato`, `tomato`  
- **Train:** 3,135 images → split 80/20 → **2,508 train / 627 val**  
- **Test:** 351 images  
- **Notes:** class imbalance (potato largest: 898; indian market smallest: 599); original images vary in size → resized to 256×256.

---

## 🎯 Approach (high level)
1. **Preprocess**: resize to 256×256, rescale pixels to \([0,1]\).  
2. **Augment (train only)**: random crop, translation (±10%), rotation (±10%), brightness/contrast adjustments.  
3. **Train & compare**: custom CNNs (simple and complex) and transfer‑learning models (VGG16, ResNet50, InceptionV3, MobileNetV2).  
4. **Evaluate**: accuracy, precision, recall, F1, confusion matrices; track experiments with MLflow and TensorBoard.  
5. **Select** model for deployment balancing accuracy, size, and inference speed.

---

## 🛠 Methodology (concise)
- **Preprocessing:** Resize → Normalize.  
- **Augmentation:** Applied only to training pipeline to improve generalization.  
- **Training setup:** Adam optimizer, categorical cross‑entropy, callbacks (ModelCheckpoint, EarlyStopping/ReduceLROnPlateau), batch size 32.  
- **Evaluation:** Report train/val/test accuracy and macro precision/recall; inspect confusion matrices for noise class performance.

---

## 🧠 Architectures (layer‑level summaries)

### Baseline (Simple) CNN
Input 256×256×3
→ Conv2D(16, 3×3, ReLU) + MaxPool
→ Flatten
→ Dense(256, ReLU)
→ Dense(4, Softmax)
**Total params:** ~67,110,596 (dense-heavy; prone to overfitting)
### Complex CNN (custom)
Input 256×256×3
→ [Conv2D(32) → ReLU → BatchNorm → MaxPool] × 1
→ [Conv2D(64) → ReLU → BatchNorm → MaxPool] × 1
→ [Conv2D(128) → ReLU → BatchNorm → MaxPool] × 1
→ [Conv2D(256) → ReLU → BatchNorm → MaxPool] × 1
→ [Conv2D(512) → ReLU → BatchNorm → GlobalAveragePooling]
→ Dense(512) → Dropout → Dense(256) → Dense(4, Softmax)
### VGG16 (Transfer Learning)
Pretrained VGG16 (ImageNet, frozen or partially trainable)
→ GlobalAveragePooling2D
→ Dense(256, ReLU) + Dropout
→ Dense(4, Softmax)
### ResNet50 (Transfer Learning)
Pretrained ResNet50 (ImageNet)
→ GlobalAveragePooling2D
→ Dense(256, ReLU) + BatchNorm + Dropout
→ Dense(4, Softmax)
### InceptionV3 (Transfer Learning)
Pretrained InceptionV3 (ImageNet)
→ GlobalAveragePooling2D
→ Dense(256, ReLU) + Dropout
→ Dense(4, Softmax)
### MobileNetV2 (Transfer Learning + Augmented Head)
Pretrained MobileNetV2 (ImageNet, partially trainable)
→ GlobalAveragePooling2D
→ Dense(512, ReLU) + BatchNorm + Dropout
→ Dense(256, ReLU) + BatchNorm
→ Dense(4, Softmax)

## 📊 Results & Model Comparison (with augmentation)
> **Test accuracies (augmentation applied)** — reported from experiments:

| Model                      | Test Accuracy |
|---------------------------:|--------------:|
| **MobileNetV2 + Augmentation** | **96.01%** |
| InceptionV3 + Augmentation     | 93.16%     |
| Complex CNN + Augmentation     | 90.31%     |
| VGG16 + Augmentation           | 90.03%     |
| ResNet50 + Augmentation        | 71.79%     |

**Key observation:** data augmentation substantially improved robustness; MobileNetV2 achieved the best test accuracy and is the best deployment candidate due to compact size and inference speed.

---

## 🔍 Outcomes & Insights
- **Best performer:** MobileNetV2 + augmentation (96.01% test accuracy) — best trade‑off between accuracy, model size, and inference latency.  
- **Overfitting risk:** simple dense‑heavy models showed very high train accuracy but lower validation/test performance.  
- **Noise class difficulty:** models sometimes confuse potatoes with cluttered market scenes — consider multi‑scale features or attention mechanisms.  
- **Augmentation:** effective at reducing overfitting; tune augmentation strength to avoid introducing unrealistic variability.

---

## ✅ Actionable Recommendations
1. **Deploy MobileNetV2** (fine‑tuned head) for real‑time sorting on edge devices.  
2. **Improve noise filtering**: experiment with attention modules or multi‑scale fusion to separate foreground produce from cluttered backgrounds.  
3. **Address class imbalance**: use class weights, focal loss, or targeted augmentation for underrepresented classes.  
4. **Production checklist**: quantize MobileNetV2 (TFLite), add monitoring for drift, and maintain MLflow experiment logs for reproducibility.
