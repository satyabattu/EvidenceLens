# app.py – EvidenceLens ULTRA ELITE EDITION (clean rebuild, fully commented)
#
# High-level:
#   • Dark-only Streamlit UI (deep navy + cyan gradient, poster-style)
#   • YOLOv3 + optional YOLOv2 ensemble for forensic evidence detection
#   • Random Forest crime classifier (feature-vector based) with explainability
#   • Export options: PNG / JSON / CSV / optional PDF report
#   • Recent detections gallery with thumbnails + quick summary
#
# Required files in the same project directory:
#   models/evidencelens_v3.pt         -> main YOLO model (weapons, blood, tape, etc.)
#   models/evidencelens_v2.pt         -> optional YOLO model (prints only; app still works if missing)
#   crime_classifier_*.pkl            -> Random Forest model
#   feature_info_*.pkl                -> contains: {"feature_columns": [...], "label_names": [...]}
#   evidence_features.csv             -> per-object annotation counts (for accuracy calculation)
#   crime_labels.csv                  -> per-image crime_type labels (for accuracy calculation)
#
# Notes for markers:
#   • The app is self-contained and robust to missing optional files (e.g. no V2 model, no RF model).
#   • All caching is done with st.cache_resource for performance.
#   • We use width="stretch" instead of use_container_width to avoid Streamlit deprecation warnings.

import io
import json
import base64
import html
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import streamlit as st


# ======================================================
# 1. Global configuration & constants
# ======================================================

# Streamlit page configuration (title, tab icon, layout)
st.set_page_config(
    page_title="EvidenceLens",
    page_icon="🔬",
    layout="wide",
)

# Paths to YOLO models (relative to the project root)
MODEL_V3_PATH = "models/evidencelens_v3.pt"   # main model (guns, knives, blood, tape, etc.)
MODEL_V2_PATH = "models/evidencelens_v2.pt"   # optional prints model (shoeprints / fingerprints)

# Candidate file names for the Random Forest classifier + feature metadata
# The app will pick the first pair that exists.
CLASSIFIER_CANDIDATES = [
    ("crime_classifier_py311.pkl", "feature_info_py311.pkl"),
    ("crime_classifier_rf.pkl", "feature_info.pkl"),
]

# CSV files used to compute RF accuracy (dataset-level evaluation, not real-world)
FEATURES_CSV = "evidence_features.csv"
LABELS_CSV = "crime_labels.csv"

# Max number of previous scans to keep in the "Recent detections" gallery
HISTORY_LIMIT = 12


# ======================================================
# 2. Utility helpers (images, encoding, NMS, categories)
# ======================================================

def load_image(file_or_bytes) -> Image.Image:
    """
    Load a file-like object or raw bytes into a PIL Image in RGB mode.

    This is used for:
      • Uploaded images (st.file_uploader)
      • Reloading images from bytes stored in session_state
    """
    img = Image.open(file_or_bytes)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """
    Convert a PIL image into PNG bytes (for downloads or for embedding in the UI).
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pil_to_thumb_bytes(img: Image.Image, max_size: int = 220) -> bytes:
    """
    Build a small thumbnail (max dimension = max_size) and return it as PNG bytes.

    Used for:
      • Recent detections horizontal gallery thumbnails
    """
    img_copy = img.copy()
    img_copy.thumbnail((max_size, max_size))
    return pil_to_png_bytes(img_copy)


def bytes_to_base64(b: bytes) -> str:
    """
    Base64-encode raw bytes. We use this to embed images directly in HTML <img> tags.
    """
    return base64.b64encode(b).decode("utf-8")


def soft_nms_by_label(
    detections: List[Dict],
    iou_threshold: float = 0.6,
) -> List[Dict]:
    """
    Very lightweight non-max suppression by label.

    Idea:
      • If two detections of the SAME label overlap strongly (IoU >= threshold),
        we keep only the one with higher confidence.
      • This avoids duplicate overlapping boxes for the same object.

    Detection schema:
      {
        "label": str,              # unified label name (e.g. "gun", "blood")
        "conf": float,             # confidence score from YOLO
        "xyxy": [x1, y1, x2, y2],  # bounding box coordinates
        "source": "V2" | "V3"      # which model produced this box
      }
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
    kept: List[Dict] = []

    def iou(a, b) -> float:
        """Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    for d in dets:
        duplicate = False
        for k in kept:
            if d["label"] == k["label"] and iou(d["xyxy"], k["xyxy"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(d)

    return kept


def evidence_type_to_category() -> Dict[str, str]:
    """
    Map unified evidence labels to coarse forensic categories.

    These categories control:
      • The summary cards ("Weapon", "Biological", etc.)
      • The small emoji summary in the Recent detections gallery
    """
    return {
        # Weapons
        "gun": "Weapon",
        "knife": "Weapon",

        # Ballistic evidence
        "bullet_hole": "Ballistic",
        "shell_casing": "Ballistic",

        # Biological evidence
        "blood": "Biological",

        # Trace evidence
        "fingerprint": "Trace",
        "prints": "Trace",
        "gloves": "Trace",

        # Scene context
        "crime_scene_tape": "Context",
        "drink_can": "Context",

        # Background / other
        "background": "Other",
    }


def draw_boxes(image: Image.Image, detections: List[Dict]) -> Image.Image:
    """
    Overlay bounding boxes and label text onto the original image.

    For each detection we show:
      • Box colour based on category (Weapon / Ballistic / etc.)
      • Text: `<label> <confidence%> · <source model>`
    """
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    # Try to use a clean TTF font; fall back to PIL's default if not available.
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    category_colors = {
        "Weapon": (80, 200, 255),
        "Ballistic": (255, 184, 108),
        "Biological": (255, 121, 121),
        "Trace": (196, 181, 253),
        "Context": (110, 231, 183),
        "Other": (148, 163, 184),
    }

    label_to_category = evidence_type_to_category()

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        label = det["label"]
        conf = det["conf"]
        source = det.get("source", "")

        cat = label_to_category.get(label, "Other")
        color = category_colors.get(cat, (148, 163, 184))

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Compose label text (e.g. "gun 87% · V3")
        label_text = f"{label} {int(conf * 100)}% · {source}"

        # Measure and draw background rectangle for the label
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad_x, pad_y = 4, 2
        label_box = [
            x1,
            max(0, y1 - text_h - 2 * pad_y),
            x1 + text_w + 2 * pad_x,
            y1,
        ]
        draw.rectangle(label_box, fill=(15, 23, 42, 220))
        draw.text(
            (label_box[0] + pad_x, label_box[1] + pad_y),
            label_text,
            fill=(255, 255, 255),
            font=font,
        )

    return img


# ======================================================
# 3. Model & classifier loading
# ======================================================

@st.cache_resource(show_spinner="Loading YOLO models (V3 primary, V2 traces)…")
def load_yolo_models():
    """
    Lazily load YOLOv3 and YOLOv2 models once per Streamlit session.

    YOLOv3:
      • Main detector for weapons, blood, tape, etc.
    YOLOv2 (optional):
      • Additional model specialised in prints.
      • If loading fails, we simply treat it as None and continue.
    """
    from ultralytics import YOLO

    model_v3 = YOLO(MODEL_V3_PATH)
    try:
        model_v2 = YOLO(MODEL_V2_PATH)
    except Exception:
        model_v2 = None
    return model_v3, model_v2


@st.cache_resource(show_spinner="Loading Random Forest crime classifier…")
def load_classifier_and_features():
    """
    Load the Random Forest crime-type classifier and its metadata.

    Returns:
      clf                -> the RandomForest model (or None if not found)
      feature_columns    -> list of evidence-type feature names (for vector building)
      label_names        -> list of crime type labels (class order)
      accuracy           -> float in [0,1] based on your annotated dataset, or None

    Behaviour:
      • Tries each pair in CLASSIFIER_CANDIDATES and picks the first existing pair.
      • Optionally reads evidence_features.csv + crime_labels.csv to compute accuracy.
      • If anything fails, the classifier tab will still render gracefully.
    """
    import os
    import pickle
    from sklearn.metrics import accuracy_score

    clf = None
    feature_columns = None
    label_names = None

    # Pick the first available (classifier, feature_info) pair
    for clf_path, info_path in CLASSIFIER_CANDIDATES:
        if os.path.exists(clf_path) and os.path.exists(info_path):
            with open(clf_path, "rb") as f:
                clf = pickle.load(f)
            with open(info_path, "rb") as f:
                info = pickle.load(f)
            feature_columns = info["feature_columns"]
            label_names = info["label_names"]
            break

    if clf is None:
        # No classifier available; the rest of the app still functions.
        return None, None, None, None

    # Compute accuracy on your labelled dataset (for display in header)
    accuracy = None
    try:
        feats = pd.read_csv(FEATURES_CSV)
        labels = pd.read_csv(LABELS_CSV)

        # Aggregate features per image
        agg_cols = [c for c in feats.columns if c not in ["image_name", "class_name"]]
        agg = feats.groupby("image_name")[agg_cols].sum().reset_index()
        merged = pd.merge(labels, agg, on="image_name", how="inner")

        X = merged[feature_columns]
        y = merged["crime_type"]

        preds = clf.predict(X)
        accuracy = float(accuracy_score(y, preds))
    except Exception:
        # Any issues reading CSV / computing accuracy: just hide the metric in header.
        accuracy = None

    return clf, feature_columns, label_names, accuracy


# ======================================================
# 4. YOLO output: label normalisation + ensemble
# ======================================================

def map_raw_label_to_unified(raw: str) -> Optional[str]:
    """
    Map raw YOLO class names (which may differ by dataset or version)
    into a unified set of labels used throughout this app.

    This lets us:
      • Group similar labels (e.g. "pistol" -> "gun")
      • Build a stable feature vector for the RF classifier
      • Use consistent categories in the UI
    """
    r = raw.lower().strip()
    mapping = {
        # Guns
        "gun": "gun",
        "pistol": "gun",
        "revolver": "gun",
        "handgun": "gun",

        # Knives
        "knife": "knife",
        "kitchen_knife": "knife",
        "blade": "knife",

        # Blood
        "blood": "blood",
        "bloodstain": "blood",

        # Bullet damage
        "bullet_hole": "bullet_hole",
        "bullet-hole": "bullet_hole",
        "bullet_hole_roboflow": "bullet_hole",

        # Shell casings
        "shell_casing": "shell_casing",
        "casing": "shell_casing",
        "cartridge": "shell_casing",

        # Fingerprints / prints
        "fingerprint": "fingerprint",
        "latent_print": "fingerprint",
        "print": "prints",
        "shoeprint": "prints",
        "footprint": "prints",
        "prints": "prints",

        # Gloves
        "glove": "gloves",
        "gloves": "gloves",

        # Crime scene tape
        "crime_scene_tape": "crime_scene_tape",
        "police_tape": "crime_scene_tape",
        "crime_tape": "crime_scene_tape",

        # Context items
        "drink_can": "drink_can",
        "can": "drink_can",

        # Background
        "background": "background",
    }
    return mapping.get(r, None)


def run_yolo_ensemble(image: Image.Image, base_threshold: float) -> List[Dict]:
    """
    Run the YOLO ensemble on a single image and return a cleaned list of detections.

    Steps:
      1. Run YOLOv3 on the image.
      2. Optionally run YOLOv2 (if available) and keep only "prints" detections.
      3. Convert raw class names to unified labels.
      4. Filter out low-confidence detections (< base_threshold).
      5. Apply light non-max suppression (by label) to remove duplicate overlapping boxes.
    """
    model_v3, model_v2 = load_yolo_models()
    from ultralytics.engine.results import Results  # noqa: F401  (type hint only)

    conf_th = float(base_threshold)
    detections: List[Dict] = []

    def extract(result, source_tag: str):
        """
        Extract a list of detection dicts from a single YOLO Results object.
        """
        if result.boxes is None or len(result.boxes) == 0:
            return []
        out = []
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < conf_th:
                continue

            label_raw = result.names.get(cls_idx, f"id_{cls_idx}")
            unified = map_raw_label_to_unified(label_raw)
            if unified is None:
                # Ignore classes we do not recognise or care about.
                continue

            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            out.append(
                {
                    "label": unified,
                    "conf": conf,
                    "xyxy": [x1, y1, x2, y2],
                    "source": source_tag,
                }
            )
        return out

    # Run YOLOv3 (main model)
    res_v3 = model_v3.predict(image, verbose=False)[0]
    detections.extend(extract(res_v3, "V3"))

    # Run YOLOv2 (optional prints model) – only keep "prints" label.
    if model_v2 is not None:
        res_v2 = model_v2.predict(image, verbose=False)[0]
        v2_raw = extract(res_v2, "V2")
        allowed_v2 = {"prints"}
        v2_filtered = [d for d in v2_raw if d["label"] in allowed_v2]
        detections.extend(v2_filtered)

    # Merge overlapping boxes per label
    detections = soft_nms_by_label(detections)
    return detections


# ======================================================
# 5. Crime classifier helpers (feature vector + RF)
# ======================================================

def build_feature_vector_from_detections(
    detections: List[Dict],
    feature_columns: List[str],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Convert detection list into a simple feature vector for the RF model.

    For each evidence type in feature_columns, we count how many objects of that
    type were detected in the image.

    Example:
        feature_columns = ["gun", "knife", "blood"]
        detections      = [gun, gun, blood]
        -> counts = {"gun": 2, "knife": 0, "blood": 1}
        -> feature vector = [2, 0, 1]
    """
    counts = {col: 0 for col in feature_columns}
    for d in detections:
        label = d["label"]
        if label in counts:
            counts[label] += 1
    vec = np.array([counts[col] for col in feature_columns])
    return vec, counts


def run_crime_classifier(detections: List[Dict]):
    """
    Run the Random Forest crime-type classifier based on detection counts.

    Returns (possibly containing None values if classifier is unavailable):
        top_label      -> predicted crime type string
        top_prob       -> float probability of top_label
        probs_table    -> list[(crime_type, prob)] sorted descending
        counts         -> dict[evidence_type -> count]
        contrib        -> list[(feature, contribution_score, count, importance)] or None
        accuracy       -> RF accuracy on your annotated dataset (float 0–1) or None
    """
    clf, feature_cols, label_names, accuracy = load_classifier_and_features()
    if clf is None or not detections or feature_cols is None:
        # No classifier / no detections / no features: gracefully skip.
        return None, None, None, None, None, accuracy

    X_vec, counts = build_feature_vector_from_detections(detections, feature_cols)
    X_df = pd.DataFrame([X_vec], columns=feature_cols)

    proba = clf.predict_proba(X_df)[0]
    top_idx = int(np.argmax(proba))
    top_label = label_names[top_idx]
    top_prob = float(proba[top_idx])

    # Build probability table for all classes with non-tiny probability.
    probs_table = []
    for idx, name in enumerate(label_names):
        p = float(proba[idx])
        if p <= 0.0005:
            continue
        probs_table.append((name, p))
    probs_table.sort(key=lambda x: x[1], reverse=True)

    # Simple contribution score:
    #   contribution = feature_importance * count
    # Gives a rough idea which evidence types pushed the model most.
    try:
        importances = clf.feature_importances_
        contrib_scores = X_vec * importances
        contrib = list(zip(feature_cols, contrib_scores, X_vec, importances))
        contrib.sort(key=lambda x: x[1], reverse=True)
    except Exception:
        contrib = None

    return top_label, top_prob, probs_table, counts, contrib, accuracy


# ======================================================
# 6. Streamlit UI helpers (session, CSS, header, history)
# ======================================================

def init_session_state():
    """
    Initialise all keys in st.session_state that this app relies on.

    This ensures:
      • Page reloads and clear actions behave predictably.
      • We can safely read these keys later without KeyError.
    """
    ss = st.session_state
    if "history" not in ss:
        ss.history = []             # list of recent scan dicts
    if "last_detection_count" not in ss:
        ss.last_detection_count = 0
    if "last_classifier_acc" not in ss:
        ss.last_classifier_acc = None
    if "detections" not in ss:
        ss.detections = []
    if "img_boxes_png" not in ss:
        ss.img_boxes_png = None
    if "original_image_png" not in ss:
        ss.original_image_png = None
    if "clf_top_label" not in ss:
        ss.clf_top_label = None
    if "clf_top_prob" not in ss:
        ss.clf_top_prob = None
    if "clf_probs_table" not in ss:
        ss.clf_probs_table = None
    if "clf_counts" not in ss:
        ss.clf_counts = None
    if "clf_contrib" not in ss:
        ss.clf_contrib = None


def apply_global_css():
    """
    Inject custom CSS for:
      • Dark background and typography
      • Hero header gradient (to match poster aesthetic)
      • Buttons, cards, recent-detections styling
      • Explainability card styling
    """
    css = """
<style>
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
body {
    background-color: #020617;
    color: #e5e7eb;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: #020617;
}

/* HEADER */
.evid-header {
    border-radius: 1.9rem;
    padding: 1.4rem 1.8rem 1.3rem 1.8rem;
    background: linear-gradient(90deg, #4f9ef8 0%, #1f4b99 30%, #020617 100%);
    box-shadow: 0 18px 50px rgba(15,23,42,0.95);
    border: 1px solid rgba(15,23,42,0.9);
    margin-bottom: 1.6rem;
}
.evid-title {
    font-size: 1.8rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: #e5f2ff;
    font-weight: 600;
}
.evid-subtitle {
    font-size: 0.82rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(226,232,240,0.9);
    margin-top: 0.45rem;
}
.evid-badges-row {
    margin-top: 0.9rem;
    display: flex;
    gap: 0.65rem;
    flex-wrap: wrap;
}
.pill-badge {
    font-size: 0.80rem;
    padding: 0.28rem 0.95rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.65);
    color: rgba(229,231,235,0.96);
    background: radial-gradient(circle at 0 0, rgba(15,23,42,0.98), rgba(15,23,42,0.90));
    box-shadow: 0 6px 20px rgba(15,23,42,0.9);
}
.pill-detect-count {
    background: linear-gradient(90deg, #4fd1ff, #6366f1);
    color: #020617;
    border: none;
    font-weight: 600;
    box-shadow: 0 18px 40px rgba(56,189,248,0.7);
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
}
.pill-accuracy {
    border: none;
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: #022c22;
    font-weight: 600;
    box-shadow: 0 18px 40px rgba(34,197,94,0.6);
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
}
@keyframes pulse-soft {
    0%   { box-shadow: 0 0 0 0 rgba(56,189,248,0.7); transform: translateY(0); }
    60%  { box-shadow: 0 0 0 16px rgba(56,189,248,0); transform: translateY(-1px); }
    100% { box-shadow: 0 0 0 0 rgba(56,189,248,0); transform: translateY(0); }
}
.pill-detect-count {
    animation: pulse-soft 2.4s infinite;
}

/* Uploader */
[data-testid="stFileUploader"] > div:first-child {
    border-radius: 1.5rem;
    padding: 1.1rem 1.2rem;
    border: 1px dashed rgba(148,163,184,0.65);
    background: radial-gradient(circle at 0 0, rgba(15,23,42,0.92), rgba(15,23,42,0.98));
    box-shadow: 0 14px 35px rgba(15,23,42,0.95);
}

/* Buttons */
.evid-primary-btn button {
    border-radius: 999px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
    border: none;
    background: linear-gradient(120deg, #38bdf8, #6366f1);
    color: #020617;
    box-shadow: 0 16px 40px rgba(56,189,248,0.5);
}
.evid-secondary-btn button {
    border-radius: 999px;
    padding: 0.5rem 1.3rem;
    font-weight: 500;
    border: 1px solid rgba(148,163,184,0.7);
    background: rgba(15,23,42,0.95);
    color: rgba(229,231,235,0.98);
}

/* Evidence summary cards */
.evid-card {
    background: radial-gradient(circle at 0 0, rgba(15,23,42,0.96), rgba(15,23,42,1));
    border-radius: 1.2rem;
    padding: 0.9rem 1.1rem;
    border: 1px solid rgba(148,163,184,0.5);
    min-width: 170px;
    box-shadow: 0 18px 40px rgba(15,23,42,0.95);
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}
.evid-card-title {
    font-size: 0.80rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}
.evid-card-value {
    font-size: 1.15rem;
    font-weight: 600;
}

/* Recent detections */
.recent-wrapper {
    border-radius: 1.3rem;
    padding: 0.85rem 1.0rem 1.0rem 1.0rem;
    border: 1px solid rgba(31,41,55,0.9);
    background: radial-gradient(circle at 0 0, rgba(15,23,42,0.96), rgba(2,6,23,1));
    box-shadow: 0 18px 45px rgba(15,23,42,0.95);
    margin-top: 0.35rem;
}
.recent-scroller {
    display: flex;
    gap: 0.9rem;
    overflow-x: auto;
    padding-bottom: 0.35rem;
}
.recent-card {
    min-width: 240px;
    max-width: 240px;
    border-radius: 1.1rem;
    background: radial-gradient(circle at 0% 0%, rgba(56,189,248,0.14), rgba(15,23,42,0.98));
    border: 1px solid rgba(56,189,248,0.45);
    padding: 0.55rem 0.6rem 0.7rem 0.6rem;
    box-shadow: 0 16px 40px rgba(15,23,42,0.98);
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}
.recent-thumb-box {
    position: relative;
    border-radius: 0.9rem;
    overflow: hidden;
    border: 1px solid rgba(15,23,42,0.9);
}
.recent-thumb-box img {
    width: 100%;
    display: block;
}
.recent-badge-left,
.recent-badge-right {
    position: absolute;
    top: 0.35rem;
    padding: 0.18rem 0.7rem;
    font-size: 0.72rem;
    border-radius: 999px;
}
.recent-badge-left {
    left: 0.35rem;
    background: linear-gradient(120deg, #22d3ee, #6366f1);
    color: #020617;
    font-weight: 600;
}
.recent-badge-right {
    right: 0.35rem;
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(148,163,184,0.85);
    color: rgba(229,231,235,0.96);
}
.recent-meta {
    font-size: 0.80rem;
    color: rgba(209,213,219,0.96);
}
.recent-meta strong {
    color: #e5e7eb;
    font-weight: 600;
}
.recent-reload-ghost {
    margin-top: 0.18rem;
    font-size: 0.80rem;
    color: rgba(148,163,184,0.95);
}

/* Explainability card */
.explain-card {
    border-radius: 1.2rem;
    padding: 0.9rem 1.0rem 0.7rem 1.0rem;
    background: radial-gradient(circle at 0% 0%, rgba(56,189,248,0.14), rgba(15,23,42,0.99));
    border: 1px solid rgba(56,189,248,0.60);
    box-shadow: 0 20px 55px rgba(15,23,42,0.98);
    margin-bottom: 0.4rem;
}
.explain-pill {
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.7);
    padding: 0.15rem 0.6rem;
    font-size: 0.78rem;
    color: #e5e7eb;
    background: rgba(15,23,42,0.9);
}
.explain-pill-main {
    border-color: rgba(56,189,248,0.85);
    background: radial-gradient(circle at 0 0, rgba(56,189,248,0.35), rgba(15,23,42,0.96));
    font-weight: 600;
}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


def header_section(total_detections: int, rf_accuracy: Optional[float]):
    """
    Render the top hero header: app title, tagline, detection count, RF accuracy.

    total_detections -> last_detection_count from session_state
    rf_accuracy      -> accuracy on annotated dataset (or None)
    """
    if rf_accuracy is None:
        acc_html = '<span class="pill-badge pill-accuracy">🎯 RF classifier accuracy: N/A</span>'
    else:
        acc_html = (
            f'<span class="pill-badge pill-accuracy">🎯 {rf_accuracy * 100:.1f}% RF '
            'classifier accuracy (dataset)</span>'
        )

    header_html = f"""
<div class="evid-header">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:1.7rem;">
    <div>
      <div class="evid-title">🧪 EVIDENCELENS</div>
      <div class="evid-subtitle">
        COMPUTER VISION FOR FORENSIC EVIDENCE &amp; CRIME CLASSIFICATION
      </div>
      <div class="evid-badges-row">
        <span class="pill-badge">⚡ AI-powered detection</span>
        <span class="pill-badge">⏱️ Near real-time analysis</span>
        <span class="pill-badge">📊 Detailed reports &amp; exports</span>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:0.55rem;">
      <span class="pill-badge pill-detect-count">
        🔍 {total_detections} evidence item(s) detected
      </span>
      {acc_html}
    </div>
  </div>
</div>
"""
    st.markdown(header_html, unsafe_allow_html=True)


def format_time_ago(iso_str: str) -> str:
    """
    Render a timestamp string as "X minute(s) ago", "Y day(s) ago", etc.

    Used for:
      • Recent detections gallery cards
    """
    try:
        ts = datetime.fromisoformat(iso_str)
    except Exception:
        return iso_str
    delta = datetime.utcnow() - ts
    seconds = int(delta.total_seconds())
    if seconds < 30:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute(s) ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour(s) ago"
    days = hours // 24
    return f"{days} day(s) ago"


def summarise_evidence_types(detections: List[Dict]) -> str:
    """
    Summarise which evidence categories are present for a given scan.

    Example output:
        "🔫 Weapon · 🩸 Biological"

    If no evidence is found, we return an empty string so the card text is not noisy.
    """
    cats_map = evidence_type_to_category()
    cat_counts: Dict[str, int] = {}
    for d in detections:
        cat = cats_map.get(d["label"], "Other")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    icon_map = {
        "Weapon": "🔫",
        "Ballistic": "💥",
        "Biological": "🩸",
        "Trace": "🧬",
        "Context": "📦",
        "Other": "📁",
    }

    if not cat_counts:
        return ""

    ordered = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    parts = [f"{icon_map.get(cat, '📁')} {cat}" for cat, _ in ordered]
    return " · ".join(parts)


def add_to_history(
    thumb_bytes: bytes,
    detections: List[Dict],
    clf_top_label: Optional[str],
    clf_top_prob: Optional[float],
):
    """
    Append a new scan result to the in-memory history for the Recent detections panel.

    Each history item stores:
      • thumb      -> base64 thumbnail
      • count      -> number of detections
      • ts         -> ISO timestamp when scan was run
      • summary    -> category summary string
      • crime_type -> predicted crime label (if available)
      • crime_prob -> predicted crime probability (if available)
    """
    item = {
        "thumb": bytes_to_base64(thumb_bytes),
        "count": len(detections),
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "summary": summarise_evidence_types(detections),
        "crime_type": clf_top_label,
        "crime_prob": clf_top_prob,
    }
    st.session_state.history.insert(0, item)
    st.session_state.history = st.session_state.history[:HISTORY_LIMIT]


def render_history_panel():
    """
    Render the horizontal Recent detections gallery.
    """
    st.subheader("Recent detections")
    history = st.session_state.history
    if not history:
        st.caption("Run an analysis to see thumbnails here.")
        return

    cards_html = '<div class="recent-wrapper"><div class="recent-scroller">'

    for item in history:
        img_b64 = item["thumb"]
        cnt = item["count"]
        ts = format_time_ago(item["ts"])
        summary = item.get("summary", "")
        crime_type = item.get("crime_type")
        crime_prob = item.get("crime_prob")

        meta_lines = []
        if summary:
            meta_lines.append(html.escape(summary))
        if crime_type is not None and crime_prob is not None:
            meta_lines.append(
                f"⚖️ Crime prediction: <strong>{html.escape(str(crime_type))}</strong> ({crime_prob*100:.1f}%)"
            )
        meta_html = "<br/>".join(meta_lines) if meta_lines else ""

        cards_html += f"""
<div class="recent-card">
  <div class="recent-thumb-box">
    <span class="recent-badge-left">🔍 {cnt} item(s)</span>
    <span class="recent-badge-right">⏱️ {ts}</span>
    <img src="data:image/png;base64,{img_b64}" />
  </div>
  <div class="recent-meta">
    {meta_html}
  </div>
  <div class="recent-reload-ghost">Reload this analysis</div>
</div>
"""

    cards_html += "</div></div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def render_contrib_chart(contrib_df: pd.DataFrame):
    """
    Horizontal bar chart for feature contributions (scaled 0–1).

    • X-axis: relative contribution (0–1, strongest feature = 1.0)
    • Y-axis: evidence types sorted from least to most contributing
    """
    if contrib_df.empty:
        return

    import matplotlib.pyplot as plt

    max_score = contrib_df["Contribution score"].max()
    if max_score <= 0:
        contrib_df["Relative"] = 0.0
    else:
        contrib_df["Relative"] = contrib_df["Contribution score"] / max_score

    contrib_df = contrib_df.sort_values("Relative", ascending=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.5))

    y = contrib_df["Evidence type"]
    x = contrib_df["Relative"]

    bars = ax.barh(y, x)
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=8.5,
        )

    # Dark theme styling for chart
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")
    for spine in ax.spines.values():
        spine.set_color("#4b5563")
    ax.tick_params(colors="#e5e7eb", labelsize=9)
    ax.set_xlabel("Relative contribution (0–1, scaled)", color="#e5e7eb", fontsize=9)
    ax.set_title("Feature contributions", color="#e5e7eb", fontsize=11, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    st.pyplot(fig)


# ======================================================
# 7. Main Streamlit app
# ======================================================

def main():
    """
    Top-level EvidenceLens Streamlit app.

    Pipeline:
      1. User uploads a crime scene image and chooses confidence threshold.
      2. YOLO ensemble detects evidence; bounding boxes are drawn.
      3. Evidence counts are turned into a feature vector.
      4. Random Forest predicts crime type + explains contributions.
      5. User can export annotated image + structured detection data + PDF report.
    """
    init_session_state()
    ss = st.session_state

    apply_global_css()

    # Pre-load classifier early, so we can show dataset accuracy in the hero header.
    clf, feature_cols, label_names, rf_accuracy = load_classifier_and_features()
    ss.last_classifier_acc = rf_accuracy

    # Header with gradient + stats
    header_section(ss.last_detection_count, rf_accuracy)

    st.markdown("### Input image")

    # Two-column layout:
    #   left  -> upload, controls, categories, recent history
    #   right -> detection / classification / exports tabs
    left_col, right_col = st.columns([0.90, 1.10])

    uploaded_image: Optional[Image.Image] = None
    run_clicked = False
    clear_clicked = False

    # -------------------------------
    # LEFT COLUMN – input & controls
    # -------------------------------
    with left_col:
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload a crime scene photo to detect forensic evidence.",
            type=["jpg", "jpeg", "png"],
        )

        # Persist uploaded image bytes in session_state so we can keep showing it
        # even if the widget "loses" its state.
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            ss.original_image_png = file_bytes
            uploaded_image = load_image(io.BytesIO(file_bytes))
        elif ss.original_image_png is not None:
            uploaded_image = load_image(io.BytesIO(ss.original_image_png))

        # Slider for confidence threshold used by YOLO models
        base_threshold = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.5,
            value=0.25,
            step=0.05,
            help="Lower values find more objects (with more false positives).",
        )

        # Action buttons: Analyse + Clear
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            st.markdown('<div class="evid-primary-btn">', unsafe_allow_html=True)
            run_clicked = st.button("🔎 Analyse evidence")
            st.markdown("</div>", unsafe_allow_html=True)
        with btn_col2:
            st.markdown('<div class="evid-secondary-btn">', unsafe_allow_html=True)
            clear_clicked = st.button("🧹 Clear")
            st.markdown("</div>", unsafe_allow_html=True)

        # Clear resets all state back to a blank app
        if clear_clicked:
            ss.history = []
            ss.last_detection_count = 0
            ss.detections = []
            ss.img_boxes_png = None
            ss.original_image_png = None
            ss.clf_top_label = None
            ss.clf_top_prob = None
            ss.clf_probs_table = None
            ss.clf_counts = None
            ss.clf_contrib = None

        # Show uploaded image preview (raw, without boxes)
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded image")

        # Quick reference for evidence categories supported by the model
        st.markdown("### Categories & examples")
        cats = [
            ("Weapon", "gun, knife"),
            ("Ballistic", "bullet hole, shell casing"),
            ("Biological", "blood"),
            ("Trace", "fingerprints, prints, gloves"),
            ("Context", "crime scene tape, drink can"),
        ]
        for name, desc in cats:
            st.markdown(f"- **{name}** – {desc}")

        # Recent detections gallery (thumbnails + summary)
        render_history_panel()

    # --------------------------------------------------
    # Run analysis pipeline when the user clicks Analyse
    # --------------------------------------------------
    if run_clicked and uploaded_image is not None:
        # YOLO ensemble (V3 primary, V2 prints)
        detections = run_yolo_ensemble(uploaded_image, base_threshold)

        # Draw bounding boxes onto a copy of the uploaded image
        img_boxes = draw_boxes(uploaded_image, detections)
        ss.img_boxes_png = pil_to_png_bytes(img_boxes)
        ss.detections = detections
        ss.last_detection_count = len(detections)

        # Random Forest crime classifier
        (
            top_label,
            top_prob,
            probs_table,
            counts,
            contrib,
            accuracy_used,
        ) = run_crime_classifier(detections)

        ss.clf_top_label = top_label
        ss.clf_top_prob = top_prob
        ss.clf_probs_table = probs_table
        ss.clf_counts = counts
        ss.clf_contrib = contrib

        # Add to in-memory history for the Recent detections carousel
        thumb_bytes = pil_to_thumb_bytes(img_boxes)
        add_to_history(thumb_bytes, detections, top_label, top_prob)

    # Local copies after any state updates above
    detections = ss.detections
    img_boxes_png = ss.img_boxes_png

    # -------------------------------
    # RIGHT COLUMN – Tabs
    # -------------------------------
    with right_col:
        tab_det, tab_cls, tab_export = st.tabs(
            ["🧬 Object detection", "⚖️ Crime classification", "📁 Exports"]
        )

        # =====================
        # Tab 1: Object detection
        # =====================
        with tab_det:
            st.subheader("Ensemble detection results (V3 primary + V2 traces)")

            # If analysis hasn't been run yet
            if img_boxes_png is None:
                if uploaded_image is not None:
                    st.info(
                        "Click **Analyse evidence** to run the ensemble detector "
                        "on the uploaded image."
                    )
                    st.image(uploaded_image, caption="Awaiting analysis…")
                else:
                    st.info("Upload an image to begin.")
            else:
                # Show the annotated image with bounding boxes
                st.image(
                    Image.open(io.BytesIO(img_boxes_png)),
                    caption="Detected evidence (bounding boxes)",
                )

            if detections:
                # Build high-level category counts for summary cards
                cats_map = evidence_type_to_category()
                cat_counts: Dict[str, int] = {}
                for d in detections:
                    cat = cats_map.get(d["label"], "Other")
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1

                st.markdown("#### Evidence summary")
                if cat_counts:
                    icons = {
                        "Weapon": "🔫",
                        "Ballistic": "💥",
                        "Biological": "🩸",
                        "Trace": "🧬",
                        "Context": "📦",
                        "Other": "📁",
                    }
                    cards_html = '<div style="display:flex;flex-wrap:wrap;gap:0.85rem;">'
                    for cat, c in cat_counts.items():
                        icon = icons.get(cat, "📁")
                        cards_html += f"""
<div class="evid-card">
  <div class="evid-card-title">{icon} {cat.upper()}</div>
  <div class="evid-card-value">{c} item(s)</div>
</div>
"""
                    cards_html += "</div>"
                    st.markdown(cards_html, unsafe_allow_html=True)

                # Detailed detections table: one row per detected object
                st.markdown("#### Detailed detections")
                rows = []
                cats_map = evidence_type_to_category()
                for d in detections:
                    label = d["label"].replace("_", " ").title()
                    category = cats_map.get(d["label"], "Other")
                    conf = round(d["conf"] * 100, 1)
                    src = d.get("source", "?")
                    x1, y1, x2, y2 = d["xyxy"]
                    rows.append(
                        {
                            "Label": label,
                            "Category": category,
                            "Confidence (%)": conf,
                            "Source": src,
                            "x1": round(x1, 1),
                            "y1": round(y1, 1),
                            "x2": round(x2, 1),
                            "y2": round(y2, 1),
                        }
                    )
                if rows:
                    det_df = pd.DataFrame(rows)
                    # Use width="stretch" (future-safe) instead of use_container_width
                    st.dataframe(det_df, width="stretch")
            else:
                # No detections for the current image
                if uploaded_image is not None:
                    st.info("No detections yet. Try lowering the confidence threshold.")
                else:
                    st.info("Upload an image and run the analysis.")

        # ==========================
        # Tab 2: Crime classification
        # ==========================
        with tab_cls:
            st.subheader("Crime classification")

            if not detections:
                st.info(
                    "Run object detection first so the classifier can use evidence counts."
                )
            elif ss.clf_top_label is None:
                st.markdown(
                    '<div class="classifier-error">Crime classifier is not available '
                    "or failed to run. Make sure the Random Forest model and feature "
                    "metadata pickle files exist in this folder.</div>",
                    unsafe_allow_html=True,
                )
            else:
                # Main prediction header
                st.markdown(
                    f"##### Predicted crime type: **{ss.clf_top_label}** · "
                    f"*model confidence {ss.clf_top_prob*100:.1f}%*"
                )

                # Evidence vector summary
                st.markdown("#### Evidence pattern (feature vector)")
                pat_rows = []
                for k, v in (ss.clf_counts or {}).items():
                    if v == 0:
                        continue
                    pat_rows.append(
                        {"Evidence type": k.replace("_", " ").title(), "Count": v}
                    )
                if pat_rows:
                    st.table(pd.DataFrame(pat_rows))
                else:
                    st.write("No non-zero evidence counts passed to classifier.")

                # Probability distribution over all crime types
                st.markdown("#### Probability distribution")
                probs_table = ss.clf_probs_table or []
                if probs_table:
                    probs_df = pd.DataFrame(
                        [
                            {"Crime type": name, "Probability": f"{p*100:.1f}%"}
                            for name, p in probs_table
                        ]
                    )
                    st.table(probs_df)
                else:
                    st.write("No probability scores available.")

                # Explainability section around feature contributions chart
                explain_html = """
<div class="explain-card">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:0.75rem;">
    <div style="display:flex;flex-direction:column;gap:0.30rem;">
      <div style="font-size:0.82rem;letter-spacing:0.18em;text-transform:uppercase;opacity:0.85;">
        Evidence contribution (explainability)
      </div>
      <div style="font-size:0.82rem;color:rgba(209,213,219,0.96);">
        Bars show each evidence type's <b>relative impact</b> on this prediction.
        Values are scaled so that the strongest contributor appears as <code>1.00</code>.
      </div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:0.25rem;">
      <span class="explain-pill explain-pill-main">Feature contributions (scaled 0–1)</span>
      <span class="explain-pill">RF evidence importance view</span>
    </div>
  </div>
</div>
"""
                st.markdown(explain_html, unsafe_allow_html=True)

                contrib = ss.clf_contrib
                if contrib is None:
                    st.write(
                        "Feature-level contributions are not available for this classifier."
                    )
                else:
                    contrib_df = pd.DataFrame(
                        [
                            {
                                "Evidence type": name.replace("_", " ").title(),
                                "Contribution score": float(score),
                                "Count": int(cnt),
                                "Feature importance": float(imp),
                            }
                            for name, score, cnt, imp in contrib
                            if cnt > 0
                        ]
                    )
                    if contrib_df.empty:
                        st.write("No non-zero evidence counts to attribute.")
                    else:
                        render_contrib_chart(contrib_df)
                        st.table(contrib_df)

                st.caption(
                    "The crime classifier is a Random Forest trained on your annotated "
                    "dataset. The accuracy in the header is measured on that dataset "
                    "and should be reported as an internal evaluation metric, "
                    "not as a real-world guarantee."
                )

        # =====================
        # Tab 3: Exports
        # =====================
        with tab_export:
            st.subheader("Exports")

            if not detections or img_boxes_png is None:
                st.info(
                    "Run an analysis first to enable exports of annotated image "
                    "and detection reports."
                )
            else:
                col_a, col_b, col_c, col_d = st.columns(4)

                # A. Download annotated PNG
                with col_a:
                    st.markdown('<div class="export-pill">', unsafe_allow_html=True)
                    st.download_button(
                        "🖼️ PNG (boxes)",
                        data=img_boxes_png,
                        file_name="evidencelens_annotated.png",
                        mime="image/png",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                # B. Download JSON detections (structured, machine-readable)
                det_export = [
                    {
                        "label": d["label"],
                        "confidence": float(d["conf"]),
                        "source": d.get("source", ""),
                        "bbox_xyxy": [float(v) for v in d["xyxy"]],
                    }
                    for d in detections
                ]
                json_bytes = json.dumps(det_export, indent=2).encode("utf-8")
                with col_b:
                    st.markdown('<div class="export-pill">', unsafe_allow_html=True)
                    st.download_button(
                        "📄 JSON",
                        data=json_bytes,
                        file_name="evidencelens_detections.json",
                        mime="application/json",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                # C. Download CSV detections (for spreadsheets / quick analysis)
                det_df_for_export = pd.DataFrame(det_export)
                csv_bytes = det_df_for_export.to_csv(index=False).encode("utf-8")
                with col_c:
                    st.markdown('<div class="export-pill">', unsafe_allow_html=True)
                    st.download_button(
                        "📊 CSV",
                        data=csv_bytes,
                        file_name="evidencelens_detections.csv",
                        mime="text/csv",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                # D. Optional PDF report (requires fpdf2, but app still works if missing)
                with col_d:
                    try:
                        from fpdf import FPDF  # type: ignore

                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", "B", 16)
                        pdf.cell(0, 10, "EvidenceLens Forensic Report", ln=1)
                        pdf.set_font("Arial", "", 11)
                        pdf.ln(3)
                        pdf.multi_cell(
                            0,
                            6,
                            "Automatically generated summary of detections and "
                            "crime classification.",
                        )
                        pdf.ln(4)
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(0, 8, "Detection summary", ln=1)
                        pdf.set_font("Arial", "", 11)

                        # Simple counts per label
                        label_counts: Dict[str, int] = {}
                        for d in detections:
                            label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1
                        for lbl, c in label_counts.items():
                            pdf.cell(
                                0,
                                6,
                                f"- {lbl.replace('_',' ').title()}: {c} item(s)",
                                ln=1,
                            )

                        # Crime classification section (if classifier worked)
                        if ss.clf_top_label is not None:
                            pdf.ln(4)
                            pdf.set_font("Arial", "B", 12)
                            pdf.cell(0, 8, "Crime classification", ln=1)
                            pdf.set_font("Arial", "", 11)
                            pdf.cell(
                                0,
                                6,
                                f"Predicted type: {ss.clf_top_label} "
                                f"({ss.clf_top_prob*100:.1f}% confidence)",
                                ln=1,
                            )

                        pdf_bytes = pdf.output(dest="S").encode("latin1")
                        st.markdown('<div class="export-pill">', unsafe_allow_html=True)
                        st.download_button(
                            "📄 PDF report",
                            data=pdf_bytes,
                            file_name="evidencelens_report.pdf",
                            mime="application/pdf",
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                    except Exception:
                        st.info(
                            "PDF export requires the `fpdf2` package. "
                            "Install it with `pip install fpdf2` to enable this option."
                        )


# Entry point
if __name__ == "__main__":
    main()
