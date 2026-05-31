"""
app.py
Run:  python app.py
Then open:  http://127.0.0.1:5000

USER APP:    http://127.0.0.1:5000/
ADMIN PANEL: http://127.0.0.1:5000/admin


CHECKS:
  1. Image AI Analyser        20 pts
  2. Barcode Scanner           5 pts
  3. IMEI Comparator           5 pts
  4. IMEI Online (Apple DB)   30 pts
  5. IMEI Offline Luhn        15 pts
  6. Model Number Validator   25 pts
  TOTAL                      100 pts

ADMIN FEATURES:
  - See every check request with timestamp, IP, location, scores
  - Clickable Google Maps link for each user location
  - See user feedback/challenges
  - Search and filter all requests
  - Summary stats at the top
"""

import os, pickle, re, sqlite3, json
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageFilter, ImageStat
from flask import (Flask, request, render_template_string,
                   send_from_directory, redirect, url_for, session)
from pathlib import Path

try:
    import requests as req_lib
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False


# CONFIGURATION  
IPHONE_THRESHOLD = 0.45
ADMIN_PASSWORD   = "macha2050"
SECRET_KEY       = "iphone-auth-secret-2025"

# PATHS
BASE_DIR      = Path(__file__).resolve().parent
PTH_PATH      = BASE_DIR / "models" / "iphone_classifier.pth"
INFO_PATH     = BASE_DIR / "models" / "iphone_classifier_info.pkl"
UPLOAD_FOLDER = BASE_DIR / "uploads"
DB_PATH       = BASE_DIR / "data" / "requests.db"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.secret_key = SECRET_KEY
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

def allowed_file(f):
    return "." in f and f.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# DATABASE
def init_db():
    con = sqlite3.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT,
            ip             TEXT,
            city           TEXT,
            country        TEXT,
            lat            REAL,
            lng            REAL,
            imei           TEXT,
            model_num      TEXT,
            score          INTEGER,
            verdict        TEXT,
            image_result   TEXT,
            online_result  TEXT,
            offline_result TEXT,
            feedback       TEXT
        )
    """)
    con.commit()
    con.close()

init_db()

def save_request(ip, lat, lng, city, country, imei, model_num,
                 score, verdict, image_result, online_result,
                 offline_result, feedback):
    con = sqlite3.connect(str(DB_PATH))
    con.execute("""
        INSERT INTO requests
        (timestamp,ip,city,country,lat,lng,imei,model_num,
         score,verdict,image_result,online_result,offline_result,feedback)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        ip, city, country, lat, lng, imei, model_num,
        score, verdict, image_result, online_result, offline_result, feedback
    ))
    con.commit()
    con.close()

def get_location_from_ip(ip):
    """Free IP geolocation no API key needed."""
    try:
        if ip in ("127.0.0.1", "::1", "localhost"):
            return "Localhost", "Local", 0.0, 0.0
        r = req_lib.get(
            f"http://ip-api.com/json/{ip}?fields=city,country,lat,lon,status",
            timeout=4)
        d = r.json()
        if d.get("status") == "success":
            return d.get("city","?"), d.get("country","?"), d.get("lat",0), d.get("lon",0)
    except Exception:
        pass
    return "Unknown", "Unknown", 0.0, 0.0

# IMAGE CLASSIFIER  (MobileNetV2)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

_clf_model  = None
_real_label = 1

def _build_mobilenet():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.last_channel, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return m.to(DEVICE)

def _load_classifier():
    global _clf_model, _real_label
    if not PTH_PATH.exists():
        print(f"WARNING: model not found at {PTH_PATH} run train.py first")
        return
    m = _build_mobilenet()
    m.load_state_dict(torch.load(str(PTH_PATH), map_location=DEVICE))
    m.eval()
    _clf_model = m
    if INFO_PATH.exists():
        with open(str(INFO_PATH), "rb") as f:
            info = pickle.load(f)
        _real_label = info.get("real_label", 1)
        print(f"Classifier loaded  val_acc={info.get('best_val_acc',0):.1%}")
    else:
        print("Classifier loaded (no info file)")

_load_classifier()

def _white_ratio(img):
    data = list(img.convert("RGB").resize((128,128)).getdata())
    return sum(1 for r,g,b in data if r>210 and g>210 and b>210) / len(data)

def _crop_region(img, top_frac, bottom_frac):
    """Crop a horizontal band from a PIL image by fraction of height."""
    from PIL import ImageEnhance as IE
    w, h   = img.size
    top    = int(h * top_frac)
    bottom = int(h * bottom_frac)
    crop   = img.crop((0, top, w, bottom))
    crop   = IE.Sharpness(crop).enhance(1.4)
    crop   = IE.Contrast(crop).enhance(1.15)
    return crop.resize((512, 512), Image.LANCZOS)

def _classify_crop(img):
    """Run classifier on one PIL image crop. Returns iPhone probability."""
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(_clf_model(tensor), dim=1)[0]
    return probs[_real_label].item()

def analyse_image(img_path):
    """
    Auto-crops camera region and logo region from the uploaded photo,
    classifies both crops, then averages the scores.
    This way the model only judges iPhone-specific features regardless
    of background, hands, or overall phone shape.
    Returns (message, css_class, score 0-20)
    """
    if _clf_model is None:
        return "Model not loaded. Run train.py first.", "result-red", 0
    try:
        if os.path.getsize(img_path) < 5000:
            return "Image too small. Upload a full-size photo.", "result-red", 0
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if w < 200 or h < 300:
            return "Resolution too low. Use a clearer photo.", "result-red", 0
        if _white_ratio(img) > 0.95:
            return "Image looks like a screenshot. Upload a real photo of the phone.", "result-red", 0

        # Auto-crop: camera module (top 0-45%) and logo region (40-78%)
        cam_crop  = _crop_region(img, 0.00, 0.45)
        logo_crop = _crop_region(img, 0.40, 0.78)

        p_cam  = _classify_crop(cam_crop)
        p_logo = _classify_crop(logo_crop)
        p      = (p_cam + p_logo) / 2.0

        print(f"[analyse_image] camera={p_cam:.1%}  logo={p_logo:.1%}  avg={p:.1%}")

        if p >= IPHONE_THRESHOLD:
            conf = "High" if p >= 0.70 else "Moderate"
            return (f"✅ Image looks like a genuine iPhone. {p:.1%} confidence ({conf}).",
                    "result-green", 20)
        elif p >= 0.45:
            return (f"⚠ Borderline result. iPhone probability: {p:.1%}. "
                    "Try a clearer back-view photo without a case.",
                    "result-red", 0)
        else:
            return (f"❌ Does not appear to be an iPhone. Probability: {p:.1%}. "
                    "May be a different brand or clone device.",
                    "result-red", 0)
    except Exception as e:
        return f"Image analysis error: {e}", "result-red", 0


# IMEI OFFLINE CHECK
def luhn_check(imei):
    digits = [int(d) for d in imei]
    total  = 0
    for i, n in enumerate(digits[::-1]):
        if i % 2 == 1:
            n *= 2
            if n > 9: n -= 9
        total += n
    return total % 10 == 0

def offline_imei_check(imei):
    if not imei:
        return "No IMEI provided", 0, False
    if not imei.isdigit():
        return "❌ INVALID IMEI. Must contain digits only.", 0, True
    if len(imei) != 15:
        return f"❌ INVALID IMEI LENGTH got {len(imei)} digits, need 15", 0, True
    if not luhn_check(imei):
        return "❌ INVALID IMEI. Checksum failed, this IMEI is not genuine.", 0, True
    tac = imei[:8]
    if not tac.startswith("35"):
        return (f"🔵 IMEI is structurally valid but the TAC code ({tac}) is not registered as an Apple device code", 5, False)
    return f"✅ IMEI structure is valid. Length, checksum and format all passed correctly.", 15, False


# APPLE MODEL NUMBER DATABASE  (comprehensive - regions, all variants)
# Source: Apple official identifiers + GSMA TAC registry
APPLE_MODELS = {
    # iPhone 11 series
    "A2111":"iPhone 11 (USA/Canada)",       "A2221":"iPhone 11 (Global)",
    "A2223":"iPhone 11 (China/Japan)",       "A2225":"iPhone 11 (Russia/CIS)",
    "A2160":"iPhone 11 Pro (USA)",           "A2215":"iPhone 11 Pro (Global)",
    "A2217":"iPhone 11 Pro (China/Japan)",
    "A2161":"iPhone 11 Pro Max (USA)",       "A2218":"iPhone 11 Pro Max (Global)",
    "A2220":"iPhone 11 Pro Max (China/Japan)",
    # iPhone 12 series
    "A2176":"iPhone 12 mini (USA)",          "A2399":"iPhone 12 mini (Global)",
    "A2398":"iPhone 12 mini (China)",        "A2400":"iPhone 12 mini (Japan)",
    "A2172":"iPhone 12 (USA)",               "A2403":"iPhone 12 (Global)",
    "A2402":"iPhone 12 (China)",             "A2404":"iPhone 12 (Japan)",
    "A2341":"iPhone 12 Pro (USA)",           "A2407":"iPhone 12 Pro (Global)",
    "A2406":"iPhone 12 Pro (China)",         "A2408":"iPhone 12 Pro (Japan)",
    "A2342":"iPhone 12 Pro Max (USA)",       "A2411":"iPhone 12 Pro Max (Global)",
    "A2410":"iPhone 12 Pro Max (China)",     "A2412":"iPhone 12 Pro Max (Japan)",
    # iPhone 13 series
    "A2481":"iPhone 13 mini (USA)",          "A2629":"iPhone 13 mini (Global)",
    "A2626":"iPhone 13 mini (China)",        "A2630":"iPhone 13 mini (Japan)",
    "A2628":"iPhone 13 mini (Korea)",
    "A2482":"iPhone 13 (USA)",               "A2634":"iPhone 13 (Global)",
    "A2631":"iPhone 13 (China)",             "A2635":"iPhone 13 (Japan)",
    "A2633":"iPhone 13 (Korea)",
    "A2483":"iPhone 13 Pro (USA)",           "A2639":"iPhone 13 Pro (Global)",
    "A2636":"iPhone 13 Pro (China)",         "A2640":"iPhone 13 Pro (Japan)",
    "A2638":"iPhone 13 Pro (Korea)",
    "A2484":"iPhone 13 Pro Max (USA)",       "A2644":"iPhone 13 Pro Max (Global)",
    "A2641":"iPhone 13 Pro Max (China)",     "A2645":"iPhone 13 Pro Max (Japan)",
    "A2643":"iPhone 13 Pro Max (Korea)",
    # iPhone 14 series
    "A2649":"iPhone 14 (USA)",               "A2882":"iPhone 14 (Global)",
    "A2881":"iPhone 14 (China)",             "A2884":"iPhone 14 (Japan)",
    "A2883":"iPhone 14 (Korea)",
    "A2632":"iPhone 14 Plus (USA)",          "A2886":"iPhone 14 Plus (Global)",
    "A2885":"iPhone 14 Plus (China)",        "A2888":"iPhone 14 Plus (Japan)",
    "A2887":"iPhone 14 Plus (Korea)",
    "A2650":"iPhone 14 Pro (USA)",           "A2890":"iPhone 14 Pro (Global)",
    "A2889":"iPhone 14 Pro (China)",         "A2892":"iPhone 14 Pro (Japan)",
    "A2891":"iPhone 14 Pro (Korea)",
    "A2651":"iPhone 14 Pro Max (USA)",       "A2894":"iPhone 14 Pro Max (Global)",
    "A2893":"iPhone 14 Pro Max (China)",     "A2896":"iPhone 14 Pro Max (Japan)",
    "A2895":"iPhone 14 Pro Max (Korea)",
    # iPhone 15 series
    "A3089":"iPhone 15 (USA)",               "A3090":"iPhone 15 (Global)",
    "A3092":"iPhone 15 (China)",             "A3091":"iPhone 15 (Japan/Korea)",
    "A3093":"iPhone 15 Plus (USA)",          "A3094":"iPhone 15 Plus (Global)",
    "A3096":"iPhone 15 Plus (China)",        "A3095":"iPhone 15 Plus (Japan/Korea)",
    "A3101":"iPhone 15 Pro (USA)",           "A3102":"iPhone 15 Pro (Global)",
    "A3104":"iPhone 15 Pro (China)",         "A3103":"iPhone 15 Pro (Japan/Korea)",
    "A3105":"iPhone 15 Pro Max (USA)",       "A3106":"iPhone 15 Pro Max (Global)",
    "A3108":"iPhone 15 Pro Max (China)",     "A3107":"iPhone 15 Pro Max (Japan/Korea)",
    "A2849":"iPhone 15 Pro Max (Global)",
    # iPhone 16 series
    "A3287":"iPhone 16 (USA)",               "A3288":"iPhone 16 (Global)",
    "A3290":"iPhone 16 (China)",             "A3289":"iPhone 16 (Japan/Korea)",
    "A3291":"iPhone 16 Plus (USA/Global)",   "A3293":"iPhone 16 Plus (China)",
    "A3292":"iPhone 16 Plus (Japan/Korea)",
    "A3294":"iPhone 16 Pro (USA)",           "A3295":"iPhone 16 Pro (Global)",
    "A3297":"iPhone 16 Pro (China)",         "A3296":"iPhone 16 Pro (Japan/Korea)",
    "A3298":"iPhone 16 Pro Max (USA)",       "A3299":"iPhone 16 Pro Max (Global)",
    "A3301":"iPhone 16 Pro Max (China)",     "A3300":"iPhone 16 Pro Max (Japan/Korea)",
    "A3408":"iPhone 16e (USA/Global)",       "A3409":"iPhone 16e (China)",
    # iPhone 17 series
    "A3399":"iPhone 17 (USA/Global)",        "A3400":"iPhone 17 (China)",
    "A3401":"iPhone 17 Plus (USA/Global)",   "A3402":"iPhone 17 Plus (China)",
    "A3403":"iPhone 17 Pro (USA/Global)",    "A3404":"iPhone 17 Pro (China)",
    "A3405":"iPhone 17 Pro Max (USA/Global)","A3406":"iPhone 17 Pro Max (China)",
    "A3407":"iPhone 17 Air (USA/Global)",    "A3410":"iPhone 17 Air (China)",
}

# Apple TAC prefixes — first 8 digits of IMEI registered to Apple iPhones
# Covers iPhone 11-17 models across regions
APPLE_TACS = {
    # iPhone 11
    "35427510","35394310","35279511","35394110","35394210",
    # iPhone 12
    "35194411","35853711","35194311","35853811","35194211",
    "35853911","35194511","35854011",
    # iPhone 13
    "35326712","35457712","35326612","35457612","35326512",
    "35457512","35326812","35457812","35565612","35565712",
    "35565812","35565912","35748112","35748212","35748312",
    "35748412",
    # iPhone 14
    "35260414","35692214","35260314","35692314","35260214",
    "35692114","35260514","35692414","35881114","35881214",
    "35881314","35881414","35979314","35979414","35979514",
    "35979614",
    # iPhone 15
    "35512615","35798415","35512715","35798515","35512515",
    "35798315","35512815","35798615","35203376","35203375",
    "35203476","35203575","35120015","35120115","35120215",
    "35120315",
    # iPhone 16
    "35681116","35880816","35681216","35880916","35681016",
    "35880716","35681316","35881016","35234516","35234616",
    "35234716","35234816","35234916","35235016","35235116",
    "35235216",
    # iPhone 17
    "35750017","35750117","35750217","35750317","35750417",
    "35750517","35750617","35750717",
}


# IMEI ONLINE CHECK  — TAC-based local lookup, no API needed
def online_imei_check(imei):
    if not imei:
        return "No IMEI provided for online check", 0, "result-grey"

    tac8 = imei[:8]

    # Check if TAC belongs to an Apple iPhone
    if tac8 in APPLE_TACS:
        return (
            f"\u2705 IMEI verified. TAC code {tac8} matches a registered iPhone device. "
            f"This IMEI belongs to a genuine iPhone.",
            30, "result-green"
        )

    # TAC not in Apple list - check if it starts with 35 (Apple manufacturer prefix)
    # but is not in our known list (could be a very new model)
    if tac8.startswith("35"):
        return (
            f"\u26a0 TAC {tac8} is not in our known iPhone device database. "
            f"our database. Please verify at checkcoverage.apple.com.",
            10, "result-orange"
        )

    # TAC does not belong to Apple at all
    return (
        f"\u274c TAC code {tac8} is not registered to Apple. "
        f"This IMEI does not belong to an iPhone.",
        0, "result-red"
    )

# MODEL NUMBER VALIDATOR  - local database
def validate_model_number(raw):

    if not raw:
        return "", 0, "result-grey"

    cleaned = (
        raw.strip()
        .upper()
        .replace(" ", "")
        .replace("-", "")
    )

    # Auto-fix
    if re.match(r"^\d{4}$", cleaned):
        cleaned = "A" + cleaned

    # Wrong format
    if not re.match(r"^A\d{4}$", cleaned):
        return (
            f"❌ Wrong format. "
            f"Model number must look like A2484 or A3101.",
            0,
            "result-red"
        )

    # Exact match in database
    if cleaned in APPLE_MODELS:
        return (
            f"✅ VALID. {cleaned} is a confirmed "
            f"{APPLE_MODELS[cleaned]}.",
            25,
            "result-green"
        )

    # Unknown but Apple-style
    return (
        f"⚠ {cleaned} is not in the local database. "
        f"Please verify online before continuing.",
        0,
        "result-orange"
    )

# BARCODE DATABASE
# Barcode database - Apple uses EAN-13 / UPC-A on iPhone boxes
# The barcode encodes the product SKU. We match by known Apple prefixes.
# Apple's registered GS1 prefixes: 0190198, 0190199, 0190200, 0194252...
# We also check if the full barcode contains a known Apple GS1 company prefix.

APPLE_GS1_PREFIXES = {
    "019",    # Apple Inc (main)
    "0190",   # Apple Inc
    "01901",  # Apple Inc
    "01902",  # Apple Inc
    "01903",  # Apple Inc
    "01904",  # Apple Inc
    "01905",  # Apple Inc
    "01906",  # Apple Inc
    "01907",  # Apple Inc
    "01908",  # Apple Inc
    "01909",  # Apple Inc
    "01940",  # Apple Inc
    "01941",  # Apple Inc
    "01942",  # Apple Inc
    "01943",  # Apple Inc
    "01944",  # Apple Inc
    "01945",  # Apple Inc
    "01946",  # Apple Inc
    "01947",  # Apple Inc
    "01948",  # Apple Inc
    "01949",  # Apple Inc
    "01950",  # Apple Inc
    "01951",  # Apple Inc
    "01952",  # Apple Inc
    "01953",  # Apple Inc
    "01954",  # Apple Inc
    "01955",  # Apple Inc
    "01956",  # Apple Inc
    "01957",  # Apple Inc
    "01958",  # Apple Inc
    "01959",  # Apple Inc
    "01960",  # Apple Inc
    "01961",  # Apple Inc
    "01962",  # Apple Inc
    "01963",  # Apple Inc
    "01964",  # Apple Inc
    "01965",  # Apple Inc
    "01966",  # Apple Inc
    "01967",  # Apple Inc
    "01968",  # Apple Inc
    "01969",  # Apple Inc
    "01970",  # Apple Inc
    "01971",  # Apple Inc
    "01972",  # Apple Inc
    "01973",  # Apple Inc
    "01974",  # Apple Inc
    "01975",  # Apple Inc
    "01976",  # Apple Inc
    "01977",  # Apple Inc
    "01978",  # Apple Inc
    "01979",  # Apple Inc
    "01980",  # Apple Inc
    "01981",  # Apple Inc
    "01982",  # Apple Inc
    "01983",  # Apple Inc
    "01984",  # Apple Inc
    "01985",  # Apple Inc
    "01986",  # Apple Inc
    "01987",  # Apple Inc
    "01988",  # Apple Inc
    "01989",  # Apple Inc
    "01990",  # Apple Inc
    "01991",  # Apple Inc
    "01992",  # Apple Inc
    "01993",  # Apple Inc
    "01994",  # Apple Inc
    "01995",  # Apple Inc
    "01996",  # Apple Inc
    "01997",  # Apple Inc
    "01998",  # Apple Inc
    "01999",  # Apple Inc
}

# Specific known iPhone barcodes (full 7-digit GS1 prefix)
IPHONE_BARCODES = {
    # iPhone 11 series
    "0190195": "iPhone 11",
    "0190196": "iPhone 11 Pro",
    "0190197": "iPhone 11 Pro Max",
    # iPhone 12 series
    "0190200": "iPhone 12",
    "0190199": "iPhone 12 Pro",
    "0190198": "iPhone 12 Pro Max",
    "0194025": "iPhone 12 mini",
    # iPhone 13 series
    "0194254": "iPhone 13",
    "0194253": "iPhone 13 Pro",
    "0194252": "iPhone 13 Pro Max",
    "0194028": "iPhone 13 mini",
    # iPhone 14 series
    "0194257": "iPhone 14",
    "0194256": "iPhone 14 Pro",
    "0194255": "iPhone 14 Pro Max",
    "0194031": "iPhone 14 Plus",
    # iPhone 15 series
    "0195951": "iPhone 15",
    "0195950": "iPhone 15 Pro",
    "0195949": "iPhone 15 Pro Max",
    "0195953": "iPhone 15 Plus",
    # iPhone 16 series
    "0197252": "iPhone 16",
    "0197251": "iPhone 16 Pro",
    "0197250": "iPhone 16 Pro Max",
    "0197254": "iPhone 16 Plus",
    "0197260": "iPhone 16e",
    # iPhone 17 series
    "0198502": "iPhone 17",
    "0198501": "iPhone 17 Pro",
    "0198500": "iPhone 17 Pro Max",
    "0198504": "iPhone 17 Plus",
    "0198506": "iPhone 17 Air",
}

def valid_ean13(barcode):
    """
    Validate EAN-13 checksum
    """

    if not barcode.isdigit():
        return False

    if len(barcode) != 13:
        return False

    digits = [int(d) for d in barcode]

    checksum = digits.pop()

    total = 0

    for i, num in enumerate(digits):

        if i % 2 == 0:
            total += num
        else:
            total += num * 3

    calc = (10 - (total % 10)) % 10

    return calc == checksum

def check_barcode(barcode):

    if not barcode:
        return "", 0, "result-grey"

    code = barcode.strip().replace(" ", "")

    # Only digits
    if not code.isdigit():
        return (
            "❌ Invalid barcode format.",
            0,
            "result-red"
        )

    # Valid lengths
    if len(code) not in [12, 13]:
        return (
            f"❌ Invalid barcode length ({len(code)} digits).",
            0,
            "result-red"
        )

    # Validate EAN13 checksum
    if len(code) == 13:
        if not valid_ean13(code):
            return (
                "❌ Invalid barcode checksum. Barcode appears fake.",
                0,
                "result-red"
            )

    # First 7 digits identify Apple SKU
    prefix7 = code[:7]

    if prefix7 in IPHONE_BARCODES:
        return (
            f"✅ Genuine Apple barcode detected. "
            f"Matched: {IPHONE_BARCODES[prefix7]}",
            5,
            "result-green"
        )

    # Apple GS1 company prefix
    if code.startswith("019"):
        return (
            "⚠ Apple GS1 prefix detected but exact iPhone model "
            "was not recognised.",
            2,
            "result-orange"
        )

    return (
        "❌ Barcode is not registered to Apple.",
        0,
        "result-red"
    )

# USER APP HTML
HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<meta name="apple-mobile-web-app-capable" content="yes">
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<title>iPhone Authenticator System</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; font-family: Arial, sans-serif; }
body::before {
    content: ""; position: fixed; inset: 0;
    background: url('/static/iphone16promax_2.png') no-repeat center center fixed;
    background-size: cover; z-index: 0;
    filter: brightness(2.2) contrast(1.25) saturate(1.2);
}
body::after {
    content: ""; position: fixed; inset: 0;
    background: rgba(0,0,0,0.08); z-index: 0; pointer-events: none;
}
.header {
    background: red; color: white; padding: 18px;
    display: flex; align-items: center; justify-content: center;
    gap: 12px; font-size: 26px; position: relative; z-index: 1;
}
.logo { height: 38px; filter: brightness(0); }
.container { max-width: 900px; margin: auto; padding: 20px; position: relative; z-index: 1; }
.section {
    background: linear-gradient(180deg,#1f2937,#111827);
    color: white; margin-bottom: 20px; padding: 18px;
    border-radius: 14px; border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 6px 14px rgba(0,0,0,0.45);
}
.advice-section {
    background: linear-gradient(180deg,#1a2a1a,#0f1a0f);
    color: white; margin-bottom: 20px; padding: 18px;
    border-radius: 14px; border: 1px solid rgba(34,197,94,0.2);
    box-shadow: 0 6px 14px rgba(0,0,0,0.45);
}
h2 { color: #93c5fd; margin-top: 0; }
h3 { color: #86efac; margin-top: 0; }
input:not([type=hidden]):not([type=file]):not([type=submit]),
textarea {
    width: 100%; padding: 12px; margin-top: 6px; margin-bottom: 10px;
    border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);
    background: #111827; color: white; font-family: Arial;
}
input[type=file] { color: white; margin-bottom: 10px; }
button { padding: 10px 14px; border-radius: 8px; border: none; cursor: pointer; margin-top: 10px; }
.primary   { background: #2563eb; color: white; font-size: 16px; padding: 14px 28px; }
.secondary { background: #16a34a; color: white; }
.result-green  { color: #22c55e; font-weight: bold; }
.result-red    { color: #ef4444; font-weight: bold; }
.result-blue   { color: #3b82f6; font-weight: bold; }
.result-orange { color: #f59e0b; font-weight: bold; }
.result-grey   { color: #9ca3af; font-weight: bold; }
.pts { display:inline-block; background:#1e3a5f; color:#93c5fd;
       font-size:12px; padding:2px 8px; border-radius:12px;
       margin-left:8px; font-weight:normal; vertical-align:middle; }
.info-box {
    background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3);
    border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;
    color: #93c5fd; font-size: 14px; line-height: 1.7;
}
.score-bar-wrap { background:#1f2937; border-radius:8px; height:20px; width:100%; margin-top:12px; overflow:hidden; }
.score-bar-fill { height:20px; border-radius:8px; transition:width 0.5s; }
.advice-item { display:flex; gap:12px; margin-bottom:16px; align-items:flex-start; }
.advice-icon { font-size:22px; min-width:34px; text-align:center; }
/*ADVANCED ADVICE ICON COLORS */

.advice-item.money .advice-icon i { color: #f59e0b; }   /* amber */
.advice-item.box .advice-icon i { color: #60a5fa; }     /* blue */
.advice-item.app .advice-icon i { color: #22c55e; }     /* green */
.advice-item.face .advice-icon i { color: #a78bfa; }    /* purple */
.advice-item.port .advice-icon i { color: #ef4444; }    /* red */
.advice-item.tools .advice-icon i { color: #f97316; }   /* orange */
.advice-item.signal .advice-icon i { color: #38bdf8; }  /* sky */
.advice-item.store .advice-icon i { color: #34d399; }   /* green */
.advice-text { color:#d1fae5; font-size:14px; line-height:1.6; }
.advice-text strong { color:#86efac; }
/* RESPONSIVE */
@media (max-width: 600px) {
    .header { font-size: 18px; padding: 14px 10px; gap: 8px; }
    .logo   { height: 28px; }
    .container { padding: 10px; }
    .section, .advice-section { padding: 14px 12px; border-radius: 10px; margin-bottom: 14px; }
    h2  { font-size: 16px; }
    h3  { font-size: 15px; }
    .pts { font-size: 11px; padding: 2px 6px; }
    input:not([type=hidden]):not([type=file]) { padding: 10px; font-size: 16px; }
    textarea { font-size: 14px; }
    button { width: 100%; font-size: 15px; padding: 12px; margin-top: 8px; }
    .primary { padding: 14px; }
    .info-box { font-size: 13px; padding: 8px 10px; }
    .advice-item { gap: 8px; }
    .advice-icon { font-size: 18px; min-width: 26px; }
    .advice-text { font-size: 13px; }
    #scanner { height: 180px; }
    .score-bar-wrap { height: 16px; }
}
@media (min-width: 601px) and (max-width: 900px) {
    .container { padding: 16px; }
    .header { font-size: 22px; }
}
</style>
</head>
<body>

<div class="header">
    <img src="/static/iphonelogo.png" class="logo">
    iPhone Authenticator System
</div>

<div class="container">
<form method="POST"
      enctype="multipart/form-data"
      id="mainForm">

<!-- 1. IMAGE -->
<div class="section">
<h2>1. Image Analyser <span class="pts">up to 20 pts</span> <span style="color:#ef4444;font-size:13px;">Required</span></h2>
<p style="color:#9ca3af;font-size:14px;">
    Take a clear photo of the <b>back of the phone</b>. The camera bump and Apple logo
    must be fully visible. Do not use a phone case.
</p>
<input type="file" name="image" accept=".jpg,.jpeg,.png,.webp">
<input type="hidden" name="saved_image_path" value="{{ image_path }}">
{% if image_path %}
    <p class="result-green">Image uploaded ✔</p>
    {% if not hard_stop and not pending %}
    <img src="{{ url_for('uploaded_file', filename=image_path) }}"
         style="width:200px;margin-top:10px;border-radius:10px;">
    {% endif %}
{% elif hard_stop %}
    <p class="result-red">❌ No image uploaded. Please upload a back photo of the phone.</p>
{% endif %}
{% if image_result and not hard_stop and not pending %}
<p class="{{ image_class }}">{{ image_result }}</p>
{% endif %}
</div>

<!-- 2. BARCODE -->
<div class="section">
<h2>2. Scan Box Barcode <span class="pts">up to 5 pts</span></h2>
<p style="color:#9ca3af;font-size:14px;">
    Use camera scan OR type the barcode manually. You only need one.
</p>

<!-- Camera scan -->
<style>
#scanner video {
    position:absolute !important;
    top:0 !important;
    left:0 !important;
    width:100% !important;
    height:100% !important;
    object-fit:cover !important;
    border-radius:10px !important;
    display:block !important;
    z-index:1 !important;
}

#scanner canvas {
    display: none !important;
}

/* Animated scanner line */
.scan-line {
    position: absolute;
    width: 100%;
    height: 3px;
    background: #22c55e;
    top: 50%;
    left: 0;
    z-index: 10;
    box-shadow: 0 0 10px #22c55e;
    animation: scanMove 2s linear infinite;
}

@keyframes scanMove {
    0% { top: 0%; }
    50% { top: calc(100% - 3px); }
    100% { top: 0%; }
}
</style>
<div id="scanner" style="
    width:100%;
    height:280px;
    background:#0f172a;
    border-radius:10px;
    position:relative;
    overflow:hidden;
">

    <!-- Message -->
    <p id="scanner_msg" style="
        color:#475569;
        font-size:14px;
        text-align:center;
        padding-top:120px;
        margin:0;
        position:relative;
        z-index:1;
    ">
        Press Start Camera Scan to activate
    </p>

    <!-- Green moving scan line -->
    <div id="scan_line" style="
        display:none;
        position:absolute;
        top:0;
        left:0;
        width:100%;
        height:4px;
        background:#22c55e;
        box-shadow:0 0 20px #22c55e, 0 0 40px #22c55e;
        z-index:999999;
        animation:scanMove 2s linear infinite;
        pointer-events:none;
    ">
    </div>

</div>
<div style="display:flex;gap:8px;margin-top:10px;margin-bottom:16px;">
    <button type="button" onclick="startScanner()"
            style="flex:1;background:#16a34a;color:white;border:none;padding:12px;
                border-radius:8px;cursor:pointer;font-size:14px;font-weight:bold;">
        <i class="fa-solid fa-camera"></i>
        Start Camera Scan
    </button>
    <button type="button" onclick="stopScanner()"
            style="flex:1;background:#374151;color:white;border:none;padding:12px;
                border-radius:8px;cursor:pointer;font-size:14px;">
        <i class="fa-solid fa-stop"></i>
        Stop Scanner
    </button>
</div>

<!-- Divider -->
<p style="color:#475569;font-size:13px;text-align:center;margin:0 0 12px;"> OR type manually </p>

<!-- Manual entry — value syncs to hidden input on every keystroke -->
<input type="text" id="barcode_manual"
       name="barcode_manual_display"
       placeholder="Type barcode number e.g. 0194252..."
       maxlength="20"
       inputmode="numeric"
       value="{{ barcode_value }}"
       oninput="this.value=this.value.replace(/[^0-9]/g,'');
                document.getElementById('barcode_value').value=this.value;"
       style="width:100%;padding:16px;font-size:16px;border-radius:10px;
              border:1px solid rgba(255,255,255,0.2);background:#1e293b;
              color:white;margin-bottom:4px;letter-spacing:2px;
              box-sizing:border-box;">


<input type="hidden" name="barcode_value" id="barcode_value" value="{{ barcode_value }}">
<div id="barcode_result_text" style="margin-top:12px;font-weight:bold;min-height:24px;">
{% if barcode_result and not hard_stop and not pending %}
    {% if '✅' in barcode_result %}<span class="result-green">{{ barcode_result }}</span>
    {% elif '⚠' in barcode_result %}<span class="result-orange">{{ barcode_result }}</span>
    {% else %}<span class="result-red">{{ barcode_result }}</span>{% endif %}
{% elif submitted and not hard_stop and not pending %}
    <span class="result-grey">No barcode scanned. Check skipped.</span>
{% endif %}
</div>
</div>

<!-- 3. IMEI COMPARATOR -->
<div class="section">
<h2>3. IMEI Comparator <span class="pts">up to 5 pts</span></h2>
<div class="info-box">
    ℹ️ A genuine iPhone shows the <b>exact same IMEI</b> in all 3 places.
    Fake phones often have mismatched or cloned IMEI numbers.<br>
    &nbsp;&nbsp;👉 <b>Box IMEI:</b> printed on the side label of the iPhone box<br>
    &nbsp;&nbsp;👉 <b>Settings IMEI:</b> Settings → General → About → IMEI<br>
    &nbsp;&nbsp;👉 <b>Dial IMEI:</b> open the Phone app, dial <b>*#06#</b>
</div>
<p>IMEI from Phone Box</p>
<input name="imei1" placeholder="Scan or type from the box side label" value="{{ imei1 }}" inputmode="numeric">
<p>IMEI from Settings</p>
<input name="imei2" placeholder="Settings → General → About → IMEI" value="{{ imei2 }}" inputmode="numeric">
<p>IMEI from Dialling *#06#</p>
<input name="imei3" placeholder="Dial *#06# and type the number shown" value="{{ imei3 }}" inputmode="numeric">
{% if imei_result and not hard_stop and not pending %}
<p class="{{ imei_class }}">{{ imei_result }}</p>
{% endif %}
</div>

<!-- 4. IMEI ONLINE -->
<div class="section">
<h2>4. IMEI Online Verification <span class="pts">up to 30 pts</span> <span style="color:#ef4444;font-size:13px;">Required</span></h2>
<div class="info-box">
    ℹ️ This checks your IMEI against our Apple device database.
    It uses the IMEI you entered above in the <i>Dial *#06#</i> field.
</div>
{% if hard_stop and online_result and not pending %}
<p class="result-red">{{ online_result }}</p>
{% elif online_result and not hard_stop %}
<p class="{{ online_class }}">{{ online_result }}</p>
{% if online_class == "result-orange" %}
<div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.4);border-radius:10px;padding:14px;margin-top:10px;">
    <p style="color:#fbbf24;margin:0 0 10px;font-size:14px;">
        Our database could not confirm this IMEI. Please check it yourself online then answer below.
    </p>
    <a href="https://www.imei.info/" target="_blank"
       style="display:inline-block;background:#2563eb;color:white;padding:8px 16px;
              border-radius:8px;text-decoration:none;font-size:14px;margin-bottom:6px;margin-right:8px;">
        🔗 Check on imei.info
    </a>
    <a href="https://www.imeicheck.net/" target="_blank"
       style="display:inline-block;background:#16a34a;color:white;padding:8px 16px;
              border-radius:8px;text-decoration:none;font-size:14px;margin-bottom:12px;">
        🔗 Check on imeicheck.net
    </a>
    <p style="color:#fbbf24;font-size:14px;margin:10px 0 6px;">
        Did the website confirm this is a genuine iPhone?
    </p>
    <label style="display:block;color:white;margin-bottom:6px;">
        <input type="radio" name="imei_confirm" value="yes"
               {% if imei_confirm == "yes" %}checked{% endif %}
               style="width:auto;margin-right:8px;">
        Yes, confirmed as genuine iPhone
    </label>
    <label style="display:block;color:white;">
        <input type="radio" name="imei_confirm" value="no"
               {% if imei_confirm == "no" %}checked{% endif %}
               style="width:auto;margin-right:8px;">
        No, not confirmed
    </label>
</div>
{% endif %}
{% elif submitted and not hard_stop and not pending %}
<p class="result-grey">Enter IMEI in Step 3 to run this check.</p>
{% endif %}
<input type="hidden" name="imei_confirm" value="{{ imei_confirm }}" id="imei_confirm_hidden">
</div>

<!-- 5. IMEI OFFLINE -->
<div class="section">
<h2>5. IMEI Offline Validator <span class="pts">up to 15 pts</span></h2>
<div class="info-box">
    ℹ️ Validates the IMEI structure without internet checks the length
    (must be 15 digits), the Luhn checksum, and the TAC code
    (Apple devices always start with <b>35</b>).
</div>
{% if offline_result and not hard_stop and not pending %}
    {% if '❌' in offline_result %}<p class="result-red">{{ offline_result }}</p>
    {% elif '🔵' in offline_result %}<p class="result-blue">{{ offline_result }}</p>
    {% else %}<p class="result-green">{{ offline_result }}</p>{% endif %}
{% elif submitted and not hard_stop and not pending %}
<p class="result-grey">Enter IMEI in Step 3 to run this check.</p>
{% endif %}
</div>

<!-- 6. MODEL NUMBER -->
<div class="section">
<h2>6. Model Number Validator <span class="pts">up to 25 pts</span> <span style="color:#ef4444;font-size:13px;">Required</span></h2>

<div class="info-box">
    ℹ️ Every genuine iPhone has a unique <b>model number</b> assigned by Apple.<br><br>

    <b>How to find it:</b><br>
    &nbsp;&nbsp;1. Look at the <b>back of the phone</b> near the bottom (e.g. <i>Model A2484</i>)<br>
    &nbsp;&nbsp;2. Check the <b>box label</b> (side or back sticker)<br>
    &nbsp;&nbsp;3. Go to <b>Settings → General → About → Model Number</b><br>
    &nbsp;&nbsp;&nbsp;&nbsp;Tap once to switch to format <b>A####</b><br><br>

    <b>Format:</b> A + 4 digits (example: <b>A2484</b>, <b>A3101</b>)
</div>

<p style="margin-top:10px;">Enter Model Number (e.g. A2484)</p>

<input name="model_number"
       placeholder="A2484"
       value="{{ model_number }}"
       maxlength="5"
       style="text-transform:uppercase;">

{% if hard_stop and model_result %}
<p class="result-red">{{ model_result }}</p>

{% elif model_result %}
<p class="{{ model_class }}">{{ model_result }}</p>

{% if model_class == "result-orange" %}

<div style="
    background:rgba(245,158,11,0.10);
    border:1px solid rgba(245,158,11,0.35);
    border-radius:12px;
    padding:16px;
    margin-top:12px;
">

    <p style="
        color:#fbbf24;
        margin:0 0 12px;
        font-size:14px;
        line-height:1.6;
    ">
        This model number is not found in our internal database.<br>
        Please verify it using the official IMEI phone database below before continuing.
    </p>

    <!-- Direct verification link -->
    <a href="https://www.imei.info/phonedatabase/"
       target="_blank"
       style="
            display:inline-block;
            background:#2563eb;
            color:white;
            padding:10px 16px;
            border-radius:8px;
            text-decoration:none;
            font-size:14px;
            font-weight:bold;
            margin-bottom:14px;
       ">
        🔗 Open IMEI.info Phone Database
    </a>

    <p style="
        color:#fbbf24;
        font-size:14px;
        margin:10px 0 8px;
    ">
        👉 How to check your model number:
    </p>

    <div style="
        color:#e5e7eb;
        font-size:13px;
        line-height:1.6;
        background:rgba(255,255,255,0.03);
        padding:10px;
        border-radius:8px;
        margin-bottom:12px;
    ">
        1. Open the IMEI.info page above<br>
        2. Click the box written <b>Brand and model...</b><br>
        3. Enter your model number (e.g. A2484) in the search field<br>
        4. Press search to confirm if it exists
    </div>

    <p style="
        color:#fbbf24;
        font-size:14px;
        margin:10px 0 6px;
    ">
        After checking, confirm below:
    </p>

    <label style="
        display:block;
        color:white;
        margin-bottom:8px;
        background:rgba(255,255,255,0.03);
        padding:10px;
        border-radius:8px;
        cursor:pointer;
    ">
        <input type="radio"
               name="model_confirm"
               value="yes"
               {% if model_confirm == "yes" %}checked{% endif %}
               style="width:auto;margin-right:8px;">
        ✅ Yes, the website confirms it is valid
    </label>

    <label style="
        display:block;
        color:white;
        background:rgba(255,255,255,0.03);
        padding:10px;
        border-radius:8px;
        cursor:pointer;
    ">
        <input type="radio"
               name="model_confirm"
               value="no"
               {% if model_confirm == "no" %}checked{% endif %}
               style="width:auto;margin-right:8px;">
        ❌ No, it is not found / not valid
    </label>

</div>

{% endif %}
{% endif %}
</div>

<!-- 7. RUN CHECK -->
<div class="section">
<h2>7. Run Full Authentication Check</h2>
<button type="submit" class="primary">▶ &nbsp; Run Full Check</button>
</div>

</form>

<!-- 8. FINAL VERDICT -->
<div class="section">
<h2>8. Final Verdict</h2>
{% if hard_stop or pending %}
<p style="color:#f59e0b;font-weight:bold;font-size:16px;">
    ⚠ Please complete the required step above, then click Run Full Check again.
</p>
{% elif final_result %}
    <p class="{{ final_class }}" style="font-size:20px;">{{ final_result }}</p>
    {% if score_total is not none %}
    <p style="color:#9ca3af;font-size:14px;margin-bottom:4px;">
        Score: <b>{{ score_total }}</b> / 100
    </p>
    <div class="score-bar-wrap">
        <div class="score-bar-fill" style="width:{{ score_total }}%;
            background:{% if score_total >= 75 %}linear-gradient(90deg,#16a34a,#22c55e)
            {% elif score_total >= 60 %}linear-gradient(90deg,#b45309,#f59e0b)
            {% else %}linear-gradient(90deg,#991b1b,#ef4444){% endif %};"></div>
    </div>
    {% endif %}
{% elif submitted and (score_total is none) and not hard_stop and not pending %}
<p style="color:#f59e0b;font-weight:bold;">
    ⚠ Please answer the confirmation question above before the final result can be shown.
</p>
{% else %}
<p style="color:#9ca3af;">Complete the checks above and click Run Full Check.</p>
{% endif %}
</div>

<!-- 10. RESET -->
<div class="section">
<h2>9. Test Another Phone</h2>
<button class="secondary" type="button" onclick="window.location.href='/'">
    Run New Test
</button>
</div>

<!-- 11. BUYER'S ADVICE -->
<div class="advice-section">
<h3>Extra Tips: How to Spot a Fake iPhone Beyond This App</h3>

<p style="color:#86efac;font-size:13px;margin-bottom:20px;">
    These are physical and behavioural signs you should check with your own eyes.
</p>

<!-- 1 -->
<div class="advice-item money">
    <div class="advice-icon"><i class="fa-solid fa-money-bill-wave"></i></div>
    <div class="advice-text">
        <strong>Price too cheap? Walk away.</strong><br>
        If an "iPhone 15 Pro Max" costs TSh 200,000 or feels suspiciously discounted,
        it is fake. Check the current retail price on apple.com before any purchase.
    </div>
</div>

<!-- 2 -->
<div class="advice-item box">
    <div class="advice-icon"><i class="fa-solid fa-box-open"></i></div>
    <div class="advice-text">
        <strong>Inspect the box carefully</strong><br>
        Genuine iPhone boxes have sharp clean printing, a holographic Apple sticker,
        and the serial number must match the phone.
    </div>
</div>

<!-- 3 -->
<div class="advice-item app">
    <div class="advice-icon">
        <img src="/static/iphonelogo.png" style="width:26px;height:26px;object-fit:contain;filter:invert(1);">
    </div>
    <div class="advice-text">
        <strong>Test App Store and iMessage</strong><br>
        Real iPhones open App Store instantly. Android fakes fail here.
    </div>
</div>

<!-- 4 -->
<div class="advice-item face">
    <div class="advice-icon">
        <img src="/static/iphonelogo.png" style="width:26px;height:26px;object-fit:contain;filter:invert(1);">
    </div>
    <div class="advice-text">
        <strong>Test Face ID or Touch ID</strong><br>
        Real iPhones use Secure Enclave hardware authentication.
    </div>
</div>

<!-- 5 -->
<div class="advice-item port">
    <div class="advice-icon"><i class="fa-solid fa-bolt"></i></div>
    <div class="advice-text">
        <strong>Check charging port</strong><br>
        iPhone 15 uses USB-C. Lightning = older models only.
    </div>
</div>

<!-- 6 -->
<div class="advice-item tools">
    <div class="advice-icon"><i class="fa-solid fa-screwdriver-wrench"></i></div>
    <div class="advice-text">
        <strong>Feel build quality</strong><br>
        Real iPhones feel solid and premium. Fakes feel light and plastic.
    </div>
</div>

<!-- 7 -->
<div class="advice-item signal">
    <div class="advice-icon"><i class="fa-solid fa-signal"></i></div>
    <div class="advice-text">
        <strong>Verify with Apple</strong><br>
        Use checkcoverage.apple.com to confirm serial number.
    </div>
</div>

<!-- 8 -->
<div class="advice-item store">
    <div class="advice-icon"><i class="fa-solid fa-store"></i></div>
    <div class="advice-text">
        <strong>Buy only from authorised sellers</strong><br>
        Apple Store and official resellers are safe.
    </div>
</div>
</div>

<!-- 10. REPORT A CHALLENGE -->
<div class="section" style="margin-top:20px;">
<h2>10. Report a Challenge</h2>
<p style="color:#9ca3af;font-size:14px;">
    Experiencing a problem or have a question? Describe it here and we will look into it.
</p>
<textarea name="feedback" rows="4"
    placeholder="e.g. The image check keeps rejecting my photo, or I am not sure which IMEI to use..."
    style="font-size:14px;resize:vertical;">{{ feedback }}</textarea>
<button type="submit" form="mainform" class="secondary" style="margin-top:8px;">
    <i class="fa-solid fa-paper-plane" style="margin-right:6px;color:#60a5fa;"></i>
    Submit Feedback
</button>
</div>

</div><!-- .container -->

<script src="https://cdnjs.cloudflare.com/ajax/libs/quagga/0.12.1/quagga.min.js"></script>

```html
<script>
var _scanTimer = null;
let scannerDetected = false;
let scannerRunning = false;

// START SCANNER
function startScanner() {

    // Prevent double start
    if (scannerRunning) return;

    scannerRunning = true;
    scannerDetected = false;

    // Clear scanner preview first
    const scannerBox = document.querySelector('#scanner');
    let msg = document.getElementById("scanner_msg");

    if (msg) {
        msg.innerHTML = "Starting camera...";
        msg.style.display = "block";
    }

    // Show scanning message ONLY after clicking start
    document.getElementById("barcode_result_text").innerHTML =
        "<span style='color:#9ca3af'>📷 Camera started. Point camera at barcode...</span>";

    // Clear previous timeout
    if (_scanTimer) {
        clearTimeout(_scanTimer);
        _scanTimer = null;
    }

    Quagga.init({

        inputStream: {
            type: "LiveStream",
            target: scannerBox,

            constraints: {
                facingMode: "environment",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },

            area: {
                top: "15%",
                right: "5%",
                left: "5%",
                bottom: "15%"
            }
        },

        locator: {
            patchSize: "large",
            halfSample: false
        },

        numOfWorkers: navigator.hardwareConcurrency || 4,

        decoder: {
    readers: [
        "ean_reader",
        "ean_8_reader",
        "upc_reader",
        "upc_e_reader",
        "code_128_reader",
        "code_39_reader",
        "codabar_reader"
    ]
},

        locate: true,
        frequency: 10

    }, function(err) {

        if (err) {
            scannerRunning = false;

            document.getElementById("barcode_result_text").innerHTML =
                "<span class='result-red'>❌ Camera error. Please allow camera access.</span>";
            return;
        }

        // START CAMERA
        Quagga.start();

        const line = document.getElementById("scan_line");
        if (line) {
            line.style.display = "block";
        }

        const msg = document.getElementById("scanner_msg");
        if (msg) {
            msg.style.display = "none";
        }

        // 15 SECOND TIMEOUT
        _scanTimer = setTimeout(function() {

            stopScannerInternal();

            // Show message inside scanner box
            let box = document.querySelector('#scanner');

            if (box) {
                box.innerHTML =
                    "<div style='display:flex;" +
                    "align-items:center;" +
                    "justify-content:center;" +
                    "height:100%;" +
                    "color:#f59e0b;" +
                    "font-size:14px;" +
                    "font-weight:bold;" +
                    "text-align:center;" +
                    "padding:20px;'>" +
                    "No barcode detected.<br>" +
                    "Try again or use manual entry below." +
                    "</div>";
            }

            document.getElementById("barcode_result_text").innerHTML =
                "<span class='result-orange'>⚠ No barcode detected after 15 seconds.</span>";

        }, 15000);

    });

    // REMOVE OLD LISTENERS
    try {
        Quagga.offDetected();
    } catch(e) {}

    // DETECTION EVENT
    Quagga.onDetected(function(data) {
        console.log(data);
        console.log("Detected:", data.codeResult.code);

        // Prevent multiple detections
        if (scannerDetected) return;

        let code = data.codeResult.code;

        // Remove spaces
        code = code.replace(/\s/g, '');

        // Accept only numeric barcodes
        if (!/^[0-9]{8,14}$/.test(code)) {
            return;
        }

        scannerDetected = true;
        const line = document.getElementById("scan_line");
        if (line) {
            line.style.display = "none";
        }
        // Vibrate on successful scan (mobile phones)
        if (navigator.vibrate) {
            navigator.vibrate(200);
        }

        // SAVE BARCODE
        document.getElementById("barcode_value").value = code;

        let manual = document.getElementById("barcode_manual");

        if (manual) {
            manual.value = code;
        }

        // STOP CAMERA + GREEN LINE IMMEDIATELY
        stopScannerInternal();

        // SHOW SUCCESS MESSAGE
        document.getElementById("barcode_result_text").innerHTML =
            "<span class='result-green'>" +
            "✅ Barcode detected successfully: <b>" + code + "</b><br>" +
            "Click <b>Run Full Check</b> below to validate it." +
            "</span>";

        // Replace scanner area with success message
        let box = document.querySelector('#scanner');

        if (box) {
            box.innerHTML =
                "<div style='display:flex;" +
                "align-items:center;" +
                "justify-content:center;" +
                "height:100%;" +
                "color:#22c55e;" +
                "font-size:16px;" +
                "font-weight:bold;" +
                "text-align:center;" +
                "padding:20px;'>" +
                "✅ Barcode Detected Successfully" +
                "</div>";
        }

    });
}

// INTERNAL STOP
function stopScannerInternal() {

    scannerRunning = false;

    // Stop timeout
    if (_scanTimer) {
        clearTimeout(_scanTimer);
        _scanTimer = null;
    }

    try {
        Quagga.stop();
    } catch(e) {}

    const line = document.getElementById("scan_line");
    if (line) {
        line.style.display = "none";
    }

    const msg = document.getElementById("scanner_msg");
    if (msg) {
        msg.style.display = "block";
    }
}

// MANUAL STOP BUTTON
function stopScanner() {

    stopScannerInternal();

    document.getElementById("barcode_result_text").innerHTML =
        "<span class='result-orange'>" +
        "⚠ Scanner stopped. You can use manual barcode entry below." +
        "</span>";

    let box = document.querySelector('#scanner');

    if (box) {
        box.innerHTML =
            "<div style='display:flex;" +
            "align-items:center;" +
            "justify-content:center;" +
            "height:100%;" +
            "color:#f59e0b;" +
            "font-size:14px;" +
            "font-weight:bold;" +
            "text-align:center;" +
            "padding:20px;'>" +
            "Scanner stopped." +
            "</div>";
    }
}
</script>

<script>
window.onload = function () {

    let saved = "{{ barcode_value }}";

    if (saved && saved !== "None" && saved !== "") {

        document.getElementById("barcode_value").value = saved;

        let manual = document.getElementById("barcode_manual");

        if (manual) {
            manual.value = saved;
        }

        let resultEl = document.getElementById("barcode_result_text");

        // Only show saved message if backend didn't already show one
        if (!resultEl.innerHTML.trim()) {

            resultEl.innerHTML =
                "<span style='color:#9ca3af'>" +
                "Barcode saved: <b>" + saved + "</b><br>" +
                "Click Run Full Check to validate it." +
                "</span>";
        }
    }
};
window.addEventListener("beforeunload", function () {
    try {
        Quagga.stop();
    } catch(e) {}
});
</script>
<script>
window.addEventListener("load", function () {

    // If server says hard stop happened
    const hardStop = "{{ 'true' if hard_stop else 'false' }}";

    if (hardStop !== "true") return;

    // Find first red error block
    const red = document.querySelector(
        ".result-red"
    );

    if (red) {

        // Smooth scroll to error
        red.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

        // Highlight effect
        red.style.transition = "all 0.4s ease";
        red.style.boxShadow =
            "0 0 0 4px rgba(239,68,68,0.35)";

        setTimeout(() => {
            red.style.boxShadow = "none";
        }, 2500);
    }
});
</script>
</body>
</html>
"""

# ADMIN LOGIN HTML
ADMIN_LOGIN_HTML = """
<!DOCTYPE html><html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Admin Login</title>
<style>
*{box-sizing:border-box;}
body{margin:0;font-family:Arial;background:#0f172a;display:flex;
     align-items:center;justify-content:center;min-height:100vh;}
.card{background:#1e293b;padding:36px 30px;border-radius:14px;
      width:100%;max-width:360px;box-shadow:0 8px 24px rgba(0,0,0,0.5);}
h2{color:#93c5fd;margin:0 0 20px;}
input{width:100%;padding:12px;margin-bottom:14px;border-radius:8px;
      border:1px solid rgba(255,255,255,0.1);background:#0f172a;
      color:white;font-size:15px;}
button{width:100%;padding:13px;background:#2563eb;color:white;
       border:none;border-radius:8px;font-size:16px;cursor:pointer;}
.err{color:#f87171;font-size:14px;margin-bottom:10px;}
</style></head>
<body>
<div class="card">
  <h2>🔐 Admin Login</h2>
  {% if error %}<p class="err">{{ error }}</p>{% endif %}
  <form method="POST">
    <input type="password" name="password" placeholder="Enter admin password" autofocus>
    <button type="submit">Login</button>
  </form>
</div>
</body></html>
"""

# ADMIN DASHBOARD HTML
ADMIN_HTML = """
<!DOCTYPE html><html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Admin Dashboard</title>
<style>
*{box-sizing:border-box;}
body{margin:0;font-family:Arial;background:#0f172a;color:white;}
.topbar{background:#1e1e1e;padding:14px 20px;display:flex;
        align-items:center;justify-content:space-between;
        border-bottom:1px solid rgba(255,255,255,0.08);flex-wrap:wrap;gap:10px;}
.topbar h1{margin:0;font-size:20px;color:#93c5fd;}
.logout{color:#f87171;text-decoration:none;font-size:14px;padding:6px 12px;
        border:1px solid #f87171;border-radius:6px;}
.container{padding:20px;max-width:1500px;margin:auto;}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px;}
.stat-card{background:#1e293b;border-radius:12px;padding:16px 20px;
           flex:1;min-width:130px;text-align:center;}
.stat-num{font-size:28px;font-weight:bold;color:#38bdf8;}
.stat-label{font-size:12px;color:#94a3b8;margin-top:4px;}
.filters{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;}
.filters input,.filters select{
    padding:8px 12px;border-radius:8px;
    border:1px solid rgba(255,255,255,0.1);
    background:#1e293b;color:white;font-size:14px;}
.filters input{flex:1;min-width:200px;}
.tbl-wrap{overflow-x:auto;border-radius:10px;
          border:1px solid rgba(255,255,255,0.07);}
table{width:100%;border-collapse:collapse;font-size:13px;min-width:900px;}
th{background:#1e293b;color:#93c5fd;padding:10px 8px;text-align:left;
   position:sticky;top:0;z-index:1;white-space:nowrap;}
td{padding:9px 8px;border-bottom:1px solid rgba(255,255,255,0.05);
   vertical-align:top;word-break:break-word;max-width:200px;}
tr:hover td{background:rgba(255,255,255,0.03);}
.badge-green{background:#14532d;color:#4ade80;padding:2px 8px;
             border-radius:10px;font-size:11px;white-space:nowrap;}
.badge-red{background:#7f1d1d;color:#f87171;padding:2px 8px;
           border-radius:10px;font-size:11px;white-space:nowrap;}
.badge-orange{background:#431407;color:#fb923c;padding:2px 8px;
              border-radius:10px;font-size:11px;white-space:nowrap;}
.feedback-cell{color:#fde68a;font-style:italic;}
.map-link{color:#38bdf8;text-decoration:none;font-size:12px;}
.no-data{color:#94a3b8;text-align:center;padding:40px;font-size:16px;}
@media(max-width:600px){
    .topbar h1{font-size:16px;}
    .stat-num{font-size:20px;}
    .stat-card{min-width:100px;padding:12px;}
}
</style>
</head>
<body>
<div class="topbar">
  <h1>📊 Admin Dashboard iPhone Authenticator</h1>
  <a href="/admin/logout" class="logout">Logout</a>
</div>
<div class="container">

  <div class="stats">
    <div class="stat-card">
      <div class="stat-num">{{ total }}</div>
      <div class="stat-label">Total Checks</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" style="color:#4ade80;">{{ likely_real }}</div>
      <div class="stat-label">Likely Original</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" style="color:#f87171;">{{ likely_fake }}</div>
      <div class="stat-label">Likely Fake</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" style="color:#fbbf24;">{{ uncertain }}</div>
      <div class="stat-label">Uncertain</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" style="color:#a78bfa;">{{ with_feedback }}</div>
      <div class="stat-label">With Feedback</div>
    </div>
  </div>

  <div class="filters">
    <input type="text" id="searchInput"
           placeholder="Search IMEI, location, verdict, feedback..."
           oninput="filterTable()">
    <select id="verdictFilter" onchange="filterTable()">
      <option value="">All Verdicts</option>
      <option value="ORIGINAL">Likely Original</option>
      <option value="FAKE">Likely Fake</option>
      <option value="UNCERTAIN">Uncertain</option>
    </select>
    <button onclick="deleteSelected()"
            style="background:#7f1d1d;color:#f87171;border:1px solid #991b1b;
                   padding:8px 14px;border-radius:8px;cursor:pointer;font-size:14px;">
      🗑 Delete Selected
    </button>
    <button onclick="deleteAll()"
            style="background:#450a0a;color:#f87171;border:1px solid #7f1d1d;
                   padding:8px 14px;border-radius:8px;cursor:pointer;font-size:14px;">
      ⚠ Delete All
    </button>
  </div>

  <div class="tbl-wrap">
  <table id="requestsTable">
    <thead>
      <tr>
        <th><input type='checkbox' id='selectAll' onchange='toggleAll(this)' style='width:auto;'></th>
        <th>#</th>
        <th>Time (UTC)</th>
        <th>IP Address</th>
        <th>Location</th>
        <th>Map</th>
        <th>IMEI</th>
        <th>Model No.</th>
        <th>Score</th>
        <th>Verdict</th>
        <th>Image Result</th>
        <th>Online IMEI</th>
        <th>Offline IMEI</th>
        <th>Feedback / Challenge</th>
      <th>Action</th>
      </tr>
    </thead>
    <tbody>
    {% for r in rows %}
    <tr id="row-{{ r['id'] }}">
      <td><input type='checkbox' class='row-check' value='{{ r["id"] }}' style='width:auto;'></td>
      <td>{{ r['id'] }}</td>
      <td style="white-space:nowrap;">{{ r['timestamp'] }}</td>
      <td>{{ r['ip'] }}</td>
      <td>{{ r['city'] }}, {{ r['country'] }}</td>
      <td>
        {% if r['lat'] and r['lat'] != 0 %}
        <a class="map-link"
           href="https://maps.google.com/?q={{ r['lat'] }},{{ r['lng'] }}"
           target="_blank">📍 Map</a>
        {% else %}—{% endif %}
      </td>
      <td>{{ r['imei'] or '—' }}</td>
      <td>{{ r['model_num'] or '—' }}</td>
      <td>
        {% if r['score'] >= 75 %}
          <span class="badge-green">{{ r['score'] }}/100</span>
        {% elif r['score'] >= 60 %}
          <span class="badge-orange">{{ r['score'] }}/100</span>
        {% else %}
          <span class="badge-red">{{ r['score'] }}/100</span>
        {% endif %}
      </td>
      <td>
        {% set v = r['verdict'] or '' %}
        {% if 'ORIGINAL' in v or ('LIKELY' in v and 'FAKE' not in v) %}
          <span class="badge-green">{{ v[:35] }}</span>
        {% elif 'FAKE' in v %}
          <span class="badge-red">{{ v[:35] }}</span>
        {% else %}
          <span class="badge-orange">{{ v[:35] }}</span>
        {% endif %}
      </td>
      <td>{{ (r['image_result'] or '—')[:70] }}</td>
      <td>{{ (r['online_result'] or '—')[:70] }}</td>
      <td>{{ (r['offline_result'] or '—')[:70] }}</td>
      <td class="feedback-cell">{{ r['feedback'] or '—' }}</td>
      <td>
        <button onclick="deleteRow({{ r['id'] }}, this)"
                style="background:#7f1d1d;color:#f87171;border:1px solid #991b1b;
                       padding:4px 10px;border-radius:6px;cursor:pointer;font-size:12px;">
          🗑 Delete
        </button>
      </td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
  {% if not rows %}
  <p class="no-data">No requests recorded yet. Requests will appear here after users run checks.</p>
  {% endif %}
  </div>

</div>
<script>
function filterTable() {
    let search  = document.getElementById("searchInput").value.toLowerCase();
    let verdict = document.getElementById("verdictFilter").value.toUpperCase();
    document.querySelectorAll("#requestsTable tbody tr").forEach(row => {
        let text = row.innerText.toLowerCase();
        let ok = (!search || text.includes(search)) &&
                 (!verdict || text.toUpperCase().includes(verdict));
        row.style.display = ok ? "" : "none";
    });
}

function toggleAll(cb) {
    document.querySelectorAll(".row-check").forEach(c => c.checked = cb.checked);
}

function deleteRow(id, btn) {
    if (!confirm("Delete this record?")) return;
    fetch("/admin/delete/" + id, {method:"POST"})
        .then(r => r.json())
        .then(d => {
            if (d.ok) {
                document.getElementById("row-" + id).remove();
            } else { alert("Delete failed."); }
        });
}

function deleteSelected() {
    let ids = [...document.querySelectorAll(".row-check:checked")].map(c => c.value);
    if (!ids.length) { alert("No rows selected."); return; }
    if (!confirm("Delete " + ids.length + " selected record(s)?")) return;
    fetch("/admin/delete-selected", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ids: ids})
    }).then(r => r.json()).then(d => {
        if (d.ok) ids.forEach(id => {
            let row = document.getElementById("row-" + id);
            if (row) row.remove();
        });
        else alert("Delete failed.");
    });
}

function deleteAll() {
    if (!confirm("Delete ALL records? This cannot be undone.")) return;
    fetch("/admin/delete-all", {method:"POST"})
        .then(r => r.json())
        .then(d => {
            if (d.ok) {
                document.querySelectorAll("#requestsTable tbody tr").forEach(r => r.remove());
            } else { alert("Delete failed."); }
        });
}
</script>
</body></html>
"""

# FLASK ROUTES USER APP
@app.route("/", methods=["GET", "POST"])
def home():
    image_result   = None;  image_class    = ""
    imei_result    = None;  imei_class     = ""
    online_result  = None;  online_class   = ""
    offline_result = None
    model_result   = None;  model_class    = ""
    barcode_result = ""
    final_result   = None;  final_class    = ""
    score_total    = None

    saved_imei1    = "";  saved_imei2    = "";  saved_imei3   = ""
    saved_barcode  = "";  saved_image    = "";  saved_model   = ""
    saved_feedback = "";  imei_confirm   = "";  model_confirm = ""
    submitted      = False;  pending        = False;  hard_stop = False

    if request.method == "POST":
        submitted = True

        # Inputs 
        file        = request.files.get("image")
        saved_image = request.form.get("saved_image_path", "")
        active_image = None

        if file and file.filename:
            if not allowed_file(file.filename):
                image_result = "❌ Only JPG / PNG / WEBP images allowed"
                image_class  = "result-red"
            else:
                img_path = UPLOAD_FOLDER / file.filename
                file.save(str(img_path))
                active_image = str(img_path)
                saved_image  = file.filename

        if not active_image and saved_image:
            candidate = UPLOAD_FOLDER / saved_image
            if candidate.exists():
                active_image = str(candidate)

        imei1         = request.form.get("imei1", "").strip()
        imei2         = request.form.get("imei2", "").strip()
        imei3         = request.form.get("imei3", "").strip()
        barcode_val   = request.form.get("barcode_value", "").strip()
        model_number  = request.form.get("model_number", "").strip().upper()
        saved_feedback = request.form.get("feedback", "").strip()

        saved_imei1   = imei1
        saved_imei2   = imei2
        saved_imei3   = imei3
        saved_barcode = barcode_val
        saved_model   = model_number
        imei_confirm  = request.form.get("imei_confirm", "").strip()
        model_confirm = request.form.get("model_confirm", "").strip()

        score = 0

        # CHECK ALL MANDATORY FIELDS FIRST 
        # Collect ALL missing required fields before returning
        # so the user sees all errors at once

        if not active_image and not image_result:
            image_result = "❌ No image uploaded. Please upload a back photo of the phone."
            image_class  = "result-red"
            hard_stop    = True

        if not imei3:
            online_result = "❌ IMEI is required. Please enter the IMEI from Dial *#06# in Step 3."
            online_class  = "result-red"
            hard_stop     = True

        if not model_number:
            model_result = "❌ Model number is required. Please enter it from the back of the phone."
            model_class  = "result-red"
            hard_stop    = True

        # If ANY mandatory field missing - return now showing all errors at once
        if hard_stop:
            final_result = None
            _save_to_db(locals())
            return _render(locals())

        # ALL MANDATORY FIELDS PRESENT — run all checks 

        # 1. IMAGE 
        if active_image:
            image_result, image_class, img_pts = analyse_image(active_image)
            score += img_pts

        # 2. BARCODE 
        if barcode_val:
            barcode_result, bc_pts, _ = check_barcode(barcode_val)
            score += bc_pts

        # 3. IMEI MATCH 
        if imei1 and imei2 and imei3:
            if imei1 == imei2 == imei3:
                imei_result = "✅ All 3 IMEI sources match."
                imei_class  = "result-green"
                score += 5
            else:
                imei_result = "❌ IMEI MISMATCH. The 3 sources do not match, which is a fake indicator."
                imei_class  = "result-red"
        elif imei3:
            imei_result = "⚠ Only one IMEI provided. Please enter all 3 for a full comparison."
            imei_class  = "result-orange"

        # 4. IMEI ONLINE 
        online_result, online_pts, online_class = online_imei_check(imei3)

        if online_class == "result-orange":
            if imei_confirm == "yes":
                online_result = "✅ Manually verified. Confirmed as genuine iPhone."
                online_class  = "result-green"
                online_pts    = 30
            elif imei_confirm == "no":
                online_result = "❌ IMEI not confirmed on Apple Coverage website."
                online_class  = "result-red"
                online_pts    = 0
            else:
                pending = True

        score += online_pts

        # 5. IMEI OFFLINE 
        offline_result, off_pts, hard_fail = offline_imei_check(imei3)
        if hard_fail:
            offline_class = "result-red"
        score += off_pts

        # 6. MODEL NUMBER
        model_result, mod_pts, model_class = validate_model_number(model_number)

        model_result, mod_pts, model_class = validate_model_number(model_number)

        # If red (not an iPhone model) - continue to final verdict, show all results

        # If model unknown → require manual verification
        if model_class == "result-orange":

            # User confirmed YES
            if model_confirm == "yes":

                model_result = (
                    "✅ Manually verified online. "
                    "Confirmed as genuine Apple model number."
                )

                model_class = "result-green"

                mod_pts = 25

            # User confirmed NO
            elif model_confirm == "no":

                model_result = (
                    "❌ Model number was NOT recognised online. "
                    "This device is likely fake."
                )

                model_class = "result-red"

                mod_pts = 0

                # FORCE FINAL FAILURE
                score = 0

            else:

                # STOP and wait for user confirmation
                pending = True

        score += mod_pts
        # FINAL VERDICT
        # If pending confirmation - show pending message, no verdict yet
        if pending:
            final_result = None
            score_total  = None
            _save_to_db(locals())
            return _render(locals())

        score_total = score

        # If any mandatory check flagged red - always FAKE regardless of total score
        mandatory_failed = (
            image_class   == "result-red" or
            online_class  == "result-red" or
            model_class   == "result-red"
        )

        if mandatory_failed:
            # Identify which one failed for a clear message
            reasons = []
            if image_class  == "result-red": reasons.append("Image")
            if online_class == "result-red": reasons.append("IMEI Online")
            if model_class  == "result-red": reasons.append("Model Number")
            final_result = (
                f"❌ FAKE. {' and '.join(reasons)} "
                f"check{'s' if len(reasons) > 1 else ''} confirmed this is not a genuine iPhone."
            )
            final_class = "result-red"
        elif score >= 90:
            final_result = "🔥 VERY LIKELY ORIGINAL iPHONE. All major checks passed."
            final_class  = "result-green"
        elif score >= 75:
            final_result = "✅ LIKELY ORIGINAL iPHONE. Most checks passed."
            final_class  = "result-green"
        elif score >= 60:
            final_result = "⚠ UNCERTAIN. Some checks failed or were skipped. Please inspect manually."
            final_class  = "result-orange"
        else:
            final_result = "❌ LIKELY FAKE. Too many checks failed or were not completed."
            final_class  = "result-red"

        _save_to_db(locals())

    return _render(locals())


def _save_to_db(v):
    try:
        ip = request.headers.get("X-Forwarded-For",
                                  request.remote_addr or "").split(",")[0].strip()
        city, country, lat, lng = get_location_from_ip(ip)
        save_request(
            ip=ip, lat=lat, lng=lng, city=city, country=country,
            imei=v.get("saved_imei3") or v.get("saved_imei1") or "",
            model_num=v.get("saved_model", ""),
            score=v.get("score_total") or 0,
            verdict=v.get("final_result") or "",
            image_result=v.get("image_result") or "",
            online_result=v.get("online_result") or "",
            offline_result=v.get("offline_result") or "",
            feedback=v.get("saved_feedback", ""),
        )
    except Exception as e:
        print(f"[DB save error] {e}")


def _render(v):
    return render_template_string(
        HTML,
        image_result   = v.get("image_result"),
        image_class    = v.get("image_class", ""),
        imei_result    = v.get("imei_result"),
        imei_class     = v.get("imei_class", ""),
        online_result  = v.get("online_result"),
        online_class   = v.get("online_class", ""),
        offline_result = v.get("offline_result"),
        model_result   = v.get("model_result"),
        model_class    = v.get("model_class", ""),
        barcode_result = v.get("barcode_result", ""),
        final_result   = v.get("final_result"),
        final_class    = v.get("final_class", ""),
        score_total    = v.get("score_total"),
        imei1          = v.get("saved_imei1", ""),
        imei2          = v.get("saved_imei2", ""),
        imei3          = v.get("saved_imei3", ""),
        barcode_value  = v.get("saved_barcode", ""),
        image_path     = v.get("saved_image", ""),
        model_number   = v.get("saved_model", ""),
        feedback       = v.get("saved_feedback", ""),
        submitted      = v.get("submitted", False),
        imei_confirm   = v.get("imei_confirm", ""),
        model_confirm  = v.get("model_confirm", ""),
        pending        = v.get("pending", False),
        hard_stop      = v.get("hard_stop", False),
    )


# FLASK ROUTES ADMIN
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        error = "Wrong password. Try again."
    return render_template_string(ADMIN_LOGIN_HTML, error=error)

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("admin_login"))

@app.route("/admin")
@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM requests ORDER BY id DESC"
    ).fetchall()
    con.close()
    rows         = [dict(r) for r in rows]
    total        = len(rows)
    likely_real  = sum(1 for r in rows if (r["score"] or 0) >= 75)
    likely_fake  = sum(1 for r in rows if (r["score"] or 0) < 60)
    uncertain    = total - likely_real - likely_fake
    with_feedback = sum(1 for r in rows if r["feedback"] and r["feedback"].strip())
    return render_template_string(
        ADMIN_HTML,
        rows=rows, total=total, likely_real=likely_real,
        likely_fake=likely_fake, uncertain=uncertain,
        with_feedback=with_feedback,
    )


# ADMIN DELETE ROUTES
@app.route("/admin/delete/<int:row_id>", methods=["POST"])
def admin_delete_one(row_id):
    if not session.get("admin"):
        return {"ok": False, "error": "not logged in"}, 401
    try:
        con = sqlite3.connect(str(DB_PATH))
        con.execute("DELETE FROM requests WHERE id = ?", (row_id,))
        con.commit()
        con.close()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

@app.route("/admin/delete-selected", methods=["POST"])
def admin_delete_selected():
    if not session.get("admin"):
        return {"ok": False, "error": "not logged in"}, 401
    try:
        data = request.get_json()
        ids  = [int(i) for i in data.get("ids", [])]
        if not ids:
            return {"ok": False, "error": "no ids"}
        con = sqlite3.connect(str(DB_PATH))
        con.executemany("DELETE FROM requests WHERE id = ?", [(i,) for i in ids])
        con.commit()
        con.close()
        return {"ok": True, "deleted": len(ids)}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

@app.route("/admin/delete-all", methods=["POST"])
def admin_delete_all():
    if not session.get("admin"):
        return {"ok": False, "error": "not logged in"}, 401
    try:
        con = sqlite3.connect(str(DB_PATH))
        con.execute("DELETE FROM requests")
        con.commit()
        con.close()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

# STATIC / UPLOADS
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(str(BASE_DIR / 'static'), filename)

if __name__ == "__main__":
    import webbrowser, threading
    url = "http://127.0.0.1:5000"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=5000, debug=False)