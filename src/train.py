"""
train.py 
Approach: BINARY CLASSIFICATION (iPhone vs Not-iPhone)


Architecture:
  MobileNetV2 (ImageNet pretrained) → fine-tuned binary head
  - Pretrained weights = already understands camera bumps, rounded corners,
    glass backs, logos - we just teach it iPhone-specific vs non-iPhone.
  - Fine-tuning only the last few layers keeps it stable with ~200 images.

Output: models/iphone_classifier.pth  +  models/iphone_classifier_info.pkl
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
from pathlib import Path

#  PATHS 
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data" / "processed"    # cropped camera+logo images from prepare_data.py
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PTH_PATH  = MODEL_DIR / "iphone_classifier.pth"
INFO_PATH = MODEL_DIR / "iphone_classifier_info.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

#SANITY CHECK 
real_dir = DATA_DIR / "real"
fake_dir = DATA_DIR / "fake"

for d in [real_dir, fake_dir]:
    if not d.exists() or not any(d.iterdir()):
        raise RuntimeError(
            f"\n❌ Missing or empty folder: {d}\n"
            "Run  python scripts/prepare_data.py  first to generate cropped training images."
        )

#  TRANSFORMS
# Training: heavy augmentation to be invariant to background/lighting
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.6, contrast=0.5, saturation=0.4, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Validation: clean centre crop only
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# DATASET 
# ImageFolder expects:  data/processed/fake/  and  data/processed/real/
# It auto-assigns labels alphabetically: fake=0, real=1
full_dataset = ImageFolder(root=str(DATA_DIR), transform=train_transform)
print(f"\nClass mapping: {full_dataset.class_to_idx}")   # should be {fake:0, real:1}

REAL_LABEL = full_dataset.class_to_idx.get("real", 1)
FAKE_LABEL = full_dataset.class_to_idx.get("fake", 0)

# Train / val split (80/20)
n_total = len(full_dataset)
n_val   = max(1, int(0.2 * n_total))
n_train = n_total - n_val
train_ds, val_ds = torch.utils.data.random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

# Apply val_transform to the val split
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label

val_ds_t = TransformSubset(val_ds, val_transform)

# Weighted sampler to handle class imbalance
labels_train = [full_dataset.targets[i] for i in train_ds.indices]
class_counts = [labels_train.count(0), labels_train.count(1)]
print(f"Train — fake: {class_counts[0]}, real: {class_counts[1]}")
weights      = [1.0 / class_counts[l] for l in labels_train]
sampler      = WeightedRandomSampler(weights, num_samples=len(labels_train), replacement=True)

train_loader = DataLoader(train_ds,  batch_size=16, sampler=sampler,   num_workers=0)
val_loader   = DataLoader(val_ds_t,  batch_size=16, shuffle=False,     num_workers=0)

# MODEL
def build_model() -> nn.Module:
    """
    MobileNetV2 with a binary classification head.
    - Freeze ALL layers first
    - Unfreeze the last 3 convolutional blocks + classifier
    - This gives us ImageNet's rich visual understanding while
      fine-tuning on iPhone-specific structural features
    """
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze everything
    for p in m.parameters():
        p.requires_grad = False

    # Unfreeze last 3 feature blocks (indices 15, 16, 17 of features)
    for block in list(m.features.children())[-3:]:
        for p in block.parameters():
            p.requires_grad = True

    # Replace classifier with binary head
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.last_channel, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 2),   # 0 = not iPhone, 1 = iPhone
    )
    return m.to(DEVICE)

model = build_model()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nTrainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")

# TRAINING 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-4, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=5, factor=0.5
)

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience   = patience
        self.best       = 0.0
        self.counter    = 0
        self.stop       = False
    def step(self, val_acc):
        if val_acc > self.best:
            self.best    = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

stopper    = EarlyStopping(patience=10)
best_acc   = 0.0
EPOCHS     = 40

print(f"\nTraining for up to {EPOCHS} epochs …\n")

for epoch in range(1, EPOCHS + 1):

    # train
    model.train()
    t_loss, t_correct, t_total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss    += loss.item()
        t_correct += (out.argmax(1) == labels).sum().item()
        t_total   += labels.size(0)

    #  validate 
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out   = model(imgs)
            loss  = criterion(out, labels)
            v_loss    += loss.item()
            v_correct += (out.argmax(1) == labels).sum().item()
            v_total   += labels.size(0)

    t_acc = t_correct / max(t_total, 1)
    v_acc = v_correct / max(v_total, 1)

    print(f"Ep {epoch:>3}/{EPOCHS}  "
          f"train_loss {t_loss/len(train_loader):.4f}  train_acc {t_acc:.2%}  |  "
          f"val_loss {v_loss/max(len(val_loader),1):.4f}  val_acc {v_acc:.2%}  "
          f"lr {optimizer.param_groups[0]['lr']:.1e}")

    if v_acc > best_acc:
        best_acc = v_acc
        torch.save(model.state_dict(), str(PTH_PATH))
        print(f"  ✅ Best model saved (val_acc={v_acc:.2%})")

    scheduler.step(v_acc)
    stopper.step(v_acc)
    if stopper.stop:
        print(f"\nEarly stopping at epoch {epoch}")
        break

# SAVE CLASS INFO 
info = {
    "class_to_idx": full_dataset.class_to_idx,
    "real_label":   REAL_LABEL,
    "fake_label":   FAKE_LABEL,
    "best_val_acc": best_acc,
}
with open(str(INFO_PATH), "wb") as f:
    pickle.dump(info, f)

print(f"\n✅  Classifier saved → {PTH_PATH}")
print(f"    Best val accuracy : {best_acc:.2%}")
print("\nNext steps:")
print("  python scripts/threshold_check.py")
print("  python app.py")

