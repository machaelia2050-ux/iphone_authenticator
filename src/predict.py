"""
predict.py

Tests a single image against the trained model.

"""

import os, sys, pickle, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from pathlib import Path


# PATHS  (script lives in src/, model lives in models/)
BASE_DIR  = Path(__file__).resolve().parent.parent
PTH_PATH  = BASE_DIR / "models" / "iphone_classifier.pth"
INFO_PATH = BASE_DIR / "models" / "iphone_classifier_info.pkl"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IPHONE_THRESHOLD = 0.70

# TRANSFORM  (must match train.py exactly)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# MODEL
def build_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.last_channel, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return m.to(DEVICE)

def load_model():
    if not PTH_PATH.exists():
        print(f"\n❌  Model not found: {PTH_PATH}")
        print("    Run  python src/train.py  first.\n")
        sys.exit(1)
    m = build_model()
    m.load_state_dict(torch.load(str(PTH_PATH), map_location=DEVICE))
    m.eval()
    return m


# PREDICT
def predict(image_path: str):
    # Load model + info 
    model = load_model()
    real_label = 1
    best_acc   = 0.0
    if INFO_PATH.exists():
        with open(str(INFO_PATH), "rb") as f:
            info = pickle.load(f)
        real_label = info.get("real_label", 1)
        best_acc   = info.get("best_val_acc", 0.0)

    # Validate file 
    path = Path(image_path)
    if not path.exists():
        print(f"\n❌  File not found: {image_path}\n")
        sys.exit(1)
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        print(f"\n❌  Unsupported file type: {path.suffix}\n")
        sys.exit(1)

    #  Run inference
    try:
        img    = Image.open(str(path)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0]
    except Exception as e:
        print(f"\n❌  Could not process image: {e}\n")
        sys.exit(1)

    real_prob = probs[real_label].item()
    fake_prob = 1.0 - real_prob

    # Verdict 
    if real_prob >= 0.88:
        verdict = "✅  LIKELY ORIGINAL iPHONE  (High confidence)"
    elif real_prob >= IPHONE_THRESHOLD:
        verdict = "✅  LIKELY ORIGINAL iPHONE  (Moderate confidence)"
    elif real_prob >= 0.50:
        verdict = "⚠   UNCERTAIN — borderline result, inspect manually"
    else:
        verdict = "❌  LIKELY FAKE / NOT AN iPHONE"

    # Bar helper 
    def bar(p, width=30):
        filled = int(p * width)
        return "█" * filled + "░" * (width - filled)

    #Print output 
    SEP = "─" * 52
    print(f"\n{SEP}")
    print(f"  iPhone Classifier — Prediction Result")
    print(f"{SEP}")
    print(f"  Image  : {path.name}")
    print(f"  Model  : iphone_classifier.pth  (val_acc={best_acc:.1%})")
    print(f"  Device : {DEVICE}")
    print(f"{SEP}")
    print(f"  iPhone probability     : {real_prob:.1%}  {bar(real_prob)}")
    print(f"  Non-iPhone probability : {fake_prob:.1%}  {bar(fake_prob)}")
    print(f"{SEP}")
    print(f"  {verdict}")
    print(f"{SEP}\n")

# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run iPhone authenticity prediction on a single image."
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to the phone image (JPG / PNG / WEBP)"
    )
    args = parser.parse_args()
    image_path = args.image or input("  Enter image path: ").strip().strip('"')
    predict(image_path)