"""
prepare_data.py  -  Dataset Preparer (SAFE MODE)

SUPPORTED FORMATS:
    .jpg .jpeg .png .webp .bmp .gif .tiff .tif .avif .heic .heif

WHAT IT DOES:
    1. Reads every image from data/raw/real/ and data/raw/fake/
    2. Crops camera region (top 0-45%) and logo region (40-78%)
    3. Saves crops to data/processed/real/ and data/processed/fake/
    4. Never touches or deletes original files

"""

import hashlib
from pathlib import Path
from PIL import Image, ImageEnhance

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False


# PATHS

BASE_DIR  = Path(__file__).resolve().parent.parent
RAW_REAL  = BASE_DIR / "data" / "raw"       / "real"
RAW_FAKE  = BASE_DIR / "data" / "raw"       / "fake"
PROC_REAL = BASE_DIR / "data" / "processed" / "real"
PROC_FAKE = BASE_DIR / "data" / "processed" / "fake"

for d in [RAW_REAL, RAW_FAKE, PROC_REAL, PROC_FAKE]:
    d.mkdir(parents=True, exist_ok=True)

# All supported image formats — originals are never deleted
VALID_EXTS = {
    ".jpg", ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tiff", ".tif",
    ".avif",
    ".heic", ".heif",
}

OUTPUT_SIZE = 512   # crop output size in pixels

# Crop fractions of image height
CAMERA_TOP    = 0.00   # camera module starts at top
CAMERA_BOTTOM = 0.45   # camera module ends at 45%
LOGO_TOP      = 0.40   # logo starts at 40%
LOGO_BOTTOM   = 0.78   # logo ends at 78%


def file_hash(fpath):
    h = hashlib.md5()
    with open(str(fpath), "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def open_image(fpath):
    """Open image safely, handles all formats including AVIF/HEIC."""
    try:
        img = Image.open(str(fpath))
        # Convert palette/RGBA/grayscale to RGB
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            return background
        return img.convert("RGB")
    except Exception as e:
        return None


def crop_region(img, top_frac, bottom_frac):
    """Crop and enhance a region of the image."""
    w, h    = img.size
    top     = int(h * top_frac)
    bottom  = int(h * bottom_frac)
    # Ensure minimum crop height
    if bottom - top < 50:
        bottom = top + 50
    cropped = img.crop((0, top, w, bottom))
    cropped = ImageEnhance.Sharpness(cropped).enhance(1.4)
    cropped = ImageEnhance.Contrast(cropped).enhance(1.15)
    cropped = cropped.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
    return cropped


def process_folder(raw_dir, out_dir, label):
    # Find all image files regardless of extension case
    files = []
    for f in sorted(raw_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in VALID_EXTS:
            files.append(f)

    print(f"\n{'─'*60}")
    print(f"  {label.upper():5s}  |  {len(files)} original photos in {raw_dir.name}/")
    print(f"{'─'*60}")

    if not files:
        print(f"  Folder is empty. Add photos to: {raw_dir}")
        return 0

    # Clear processed output folder (keeps originals safe)
    cleared = 0
    for old in out_dir.glob("*.jpg"):
        old.unlink()
        cleared += 1
    if cleared:
        print(f"  Cleared {cleared} old processed crops.")

    accepted   = 0
    rejected   = []
    seen_hashes = set()
    idx        = 0

    iter_files = tqdm(files, desc=f"  processing {label}") if TQDM else files

    for fpath in iter_files:
        # Duplicate check
        try:
            fh = file_hash(fpath)
        except Exception:
            rejected.append((fpath.name, "cannot read file"))
            continue

        if fh in seen_hashes:
            rejected.append((fpath.name, "duplicate skipped"))
            continue
        seen_hashes.add(fh)

        # Open image
        img = open_image(fpath)
        if img is None:
            rejected.append((fpath.name, "cannot open (corrupted or unsupported format)"))
            continue

        # Size check
        w, h = img.size
        if w < 150 or h < 200:
            rejected.append((fpath.name, f"too small ({w}x{h})"))
            continue

        # Generate and save crops
        try:
            cam_crop  = crop_region(img, CAMERA_TOP,  CAMERA_BOTTOM)
            logo_crop = crop_region(img, LOGO_TOP,    LOGO_BOTTOM)

            cam_path  = out_dir / f"camera_{label}_{idx:05d}.jpg"
            logo_path = out_dir / f"logo_{label}_{idx:05d}.jpg"

            cam_crop.save( str(cam_path),  "JPEG", quality=92, optimize=True)
            logo_crop.save(str(logo_path), "JPEG", quality=92, optimize=True)

            idx      += 1
            accepted += 1

        except Exception as e:
            rejected.append((fpath.name, f"crop failed: {e}"))
            continue

    total_crops = accepted * 2
    print(f"  Original photos used  : {accepted}")
    print(f"  Training crops saved  : {total_crops}  ({accepted} camera + {accepted} logo)")
    print(f"  Original files        : UNTOUCHED (still in {raw_dir.name}/)")

    if rejected:
        print(f"  Skipped               : {len(rejected)}")
        for name, reason in rejected[:15]:
            print(f"    x  {name:<45} ->  {reason}")
        if len(rejected) > 15:
            print(f"    ... and {len(rejected) - 15} more")

    return total_crops


def main():
    print("\n" + "=" * 60)
    print("  iPhone Dataset Preparer  (SAFE — originals never deleted)")
    print(f"  Input  : data/raw/real/  and  data/raw/fake/")
    print(f"  Output : data/processed/real/  and  data/processed/fake/")
    print(f"  Crops  : camera (0-45%)  +  logo (40-78%)  at {OUTPUT_SIZE}px")
    print("=" * 60)

    real_crops = process_folder(RAW_REAL, PROC_REAL, "real")
    fake_crops = process_folder(RAW_FAKE, PROC_FAKE, "fake")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print(f"  Real training crops  : {real_crops}")
    print(f"  Fake training crops  : {fake_crops}")
    print(f"  Total                : {real_crops + fake_crops}")
    print("=" * 60)

    issues = []
    if real_crops == 0:
        issues.append("real/ folder is empty. Add genuine iPhone back photos.")
    elif real_crops < 200:
        issues.append(f"Only {real_crops} real crops. Aim for 400+ (200 original photos).")

    if fake_crops == 0:
        issues.append("fake/ folder is empty. Add fake/Android phone back photos.")
    elif fake_crops < 200:
        issues.append(f"Only {fake_crops} fake crops. Aim for 400+ (200 original photos).")

    ratio = real_crops / max(fake_crops, 1)
    if ratio > 3.0 or ratio < 0.33:
        issues.append(
            f"Class imbalance: {real_crops} real vs {fake_crops} fake crops. "
            "Try to keep them roughly equal for best training results."
        )

    if issues:
        print("\n  Issues to fix before training:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("\n  Dataset is ready for training.")
        print(f"  Next step:  python scripts/train.py")
    print()


if __name__ == "__main__":
    main()