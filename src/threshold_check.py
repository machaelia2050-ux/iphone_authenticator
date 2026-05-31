"""
threshold_check.py

Tests the trained classifier on data/raw/real/ and data/raw/fake/.
Shows per-image probability, accuracy, and tuning advice.

RUN:
  python src/threshold_check.py
"""

import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np

BASE_DIR  = Path(__file__).resolve().parent.parent
REAL_DIR  = BASE_DIR / "data" / "raw" / "real"
FAKE_DIR  = BASE_DIR / "data" / "raw" / "fake"
PTH_PATH  = BASE_DIR / "models" / "iphone_classifier.pth"
INFO_PATH = BASE_DIR / "models" / "iphone_classifier_info.pkl"

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])

def build_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(m.last_channel, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return m.to(DEVICE)

def check_folder(folder, expected_label, name, model, real_label, threshold=0.70):
    files = [f for f in folder.glob("*") if f.suffix.lower() in VALID_EXTS]
    if not files:
        print(f"\n  WARNING: No files found in {folder}")
        return None

    print(f"\n  {name}  ({len(files)} images)")
    print(f"  {'Filename':<38} {'iPhone%':>8}  Result")
    print(f"  {'-'*52}")

    probs_all = []
    correct   = 0
    wrong     = 0

    for fpath in sorted(files)[:200]:
        try:
            img    = Image.open(fpath).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p = F.softmax(model(tensor), dim=1)[0][real_label].item()
            probs_all.append(p)
            pred = 1 if p >= threshold else 0
            ok   = (pred == expected_label)
            correct += int(ok)
            wrong   += int(not ok)
            sym = "OK" if ok else "WRONG"
            print(f"  {fpath.name[:38]:<38} {p:>7.1%}  {sym}")
        except Exception as e:
            print(f"  {fpath.name[:38]:<38}   ERROR: {e}")

    total = correct + wrong
    acc   = correct / max(total, 1)
    arr   = np.array(probs_all)
    print(f"\n  Accuracy: {correct}/{total} = {acc:.1%}")
    if len(arr):
        print(f"  iPhone prob mean={arr.mean():.1%}  "
              f"min={arr.min():.1%}  max={arr.max():.1%}")
    return acc

def main():
    for p in [PTH_PATH, INFO_PATH]:
        if not p.exists():
            sys.exit(f"Not found: {p}\nRun train.py first.")

    with open(str(INFO_PATH), "rb") as f:
        info = pickle.load(f)

    real_label = info["real_label"]
    fake_label = info["fake_label"]
    best_acc   = info.get("best_val_acc", 0)

    print("=" * 55)
    print("  THRESHOLD CHECK")
    print(f"  real_label={real_label}  fake_label={fake_label}")
    print(f"  best_val_acc (training): {best_acc:.1%}")
    print("=" * 55)

    model = build_model()
    model.load_state_dict(torch.load(str(PTH_PATH), map_location=DEVICE))
    model.eval()

    real_acc = check_folder(
        REAL_DIR, real_label, "REAL iPhones", model, real_label
    )
    fake_acc = check_folder(
        FAKE_DIR, fake_label, "FAKE / Non-iPhones", model, real_label
    )

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)

    if real_acc is not None and fake_acc is not None:
        print(f"  Real accuracy : {real_acc:.1%}")
        print(f"  Fake accuracy : {fake_acc:.1%}")
        print()
        if real_acc >= 0.90 and fake_acc >= 0.85:
            print("  Model is well calibrated. Ready to deploy.")
        else:
            print("  Model needs improvement:")
            if real_acc < 0.90:
                print("    - Too many iPhones rejected")
                print("      -> Lower IPHONE_THRESHOLD in analyse_image.py (try 0.60)")
            if fake_acc < 0.85:
                print("    - Too many fakes accepted")
                print("      -> Add more fake images and retrain")
                print("      -> Raise IPHONE_THRESHOLD in analyse_image.py (try 0.80)")

if __name__ == "__main__":
    main()