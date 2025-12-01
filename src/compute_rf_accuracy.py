import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

FEATURES_CSV = "evidence_features.csv"
LABELS_CSV = "crime_labels.csv"
CLF_PATH = "crime_classifier_py311.pkl"
INFO_PATH = "feature_info_py311.pkl"

print("Loading datasets...")

try:
    feats = pd.read_csv(FEATURES_CSV)
    labels = pd.read_csv(LABELS_CSV)
except Exception as e:
    print("❌ Could not load CSV files:", e)
    exit()

print("Records:")
print(" - Features:", len(feats))
print(" - Labels:", len(labels))

# Load classifier
try:
    with open(CLF_PATH, "rb") as f:
        clf = pickle.load(f)

    with open(INFO_PATH, "rb") as f:
        info = pickle.load(f)
except Exception as e:
    print("❌ Error loading classifier:", e)
    exit()

feature_columns = info["feature_columns"]
label_names = info["label_names"]

print("Feature columns:", feature_columns)

# Aggregate features by image
agg_cols = [c for c in feats.columns if c not in ["image_name", "class_name"]]
agg = feats.groupby("image_name")[agg_cols].sum().reset_index()

# Merge with labels
merged = pd.merge(labels, agg, on="image_name", how="inner")

print("Merged rows:", len(merged))

X = merged[feature_columns]
y = merged["crime_type"]

print("Running classifier prediction...")
preds = clf.predict(X)

acc = accuracy_score(y, preds)
print(f"\n🎯 FINAL ACCURACY: {acc * 100:.2f}%")

