"""
EvidenceLens - Crime Classification Training
Trains Random Forest classifier to predict crime types from evidence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import sys

print("=" * 80)
print("EVIDENCELENS - CRIME CLASSIFICATION TRAINING")
print("Training Random Forest to predict crime types from evidence")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

# Load evidence features
print("\n📂 Loading training data...")
evidence_df = pd.read_csv('evidence_features.csv')
crime_labels_df = pd.read_csv('crime_labels.csv')

print(f"✓ Evidence features: {len(evidence_df)} images")
print(f"✓ Crime labels: {len(crime_labels_df)} images")

# Merge evidence with crime labels
df = evidence_df.merge(crime_labels_df, on='image_name')
print(f"✓ Merged dataset: {len(df)} labeled images")

# Check class distribution
print(f"\n📊 Class Distribution:")
class_counts = df['crime_type'].value_counts()
for crime, count in class_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   {crime.upper()}: {count} images ({percentage:.1f}%)")

# Remove classes with too few samples (< 10)
MIN_SAMPLES = 10
classes_to_remove = []
for crime, count in class_counts.items():
    if count < MIN_SAMPLES:
        classes_to_remove.append(crime)
        print(f"   ⚠ {crime.upper()} has only {count} samples - REMOVING")

if classes_to_remove:
    df = df[~df['crime_type'].isin(classes_to_remove)]
    print(f"\n✓ Filtered dataset: {len(df)} images")
    print(f"✓ Training on {df['crime_type'].nunique()} crime types")

# ============================================================================
# PREPARE FEATURES
# ============================================================================

# Feature columns (all evidence types EXCEPT background)
feature_columns = [
    'blood', 'bullet_hole', 'crime_scene_tape', 'drink_can',
    'fingerprint', 'gloves', 'gun', 'knife', 'shell_casing', 'prints'
]

print(f"\n🔍 Feature Engineering:")
print(f"   Using {len(feature_columns)} evidence types as features")

# Create feature matrix
X = df[feature_columns]
y = df['crime_type']

print(f"   Feature matrix: {X.shape}")
print(f"   Target classes: {y.nunique()}")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print(f"\n📊 Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintain class distribution
)

print(f"   Training set: {len(X_train)} images")
print(f"   Test set: {len(X_test)} images")

# ============================================================================
# TRAIN RANDOM FOREST CLASSIFIER
# ============================================================================

print(f"\n🌲 Training Random Forest Classifier...")

# Initialize Random Forest with optimized parameters
rf_classifier = RandomForestClassifier(
    n_estimators=200,           # More trees = better accuracy
    max_depth=10,               # Prevent overfitting
    min_samples_split=5,        # Minimum samples to split
    min_samples_leaf=2,         # Minimum samples in leaf
    class_weight='balanced',    # Handle class imbalance
    random_state=42,
    n_jobs=-1                   # Use all CPU cores
)

# Train the model
rf_classifier.fit(X_train, y_train)
print(f"✓ Model trained successfully!")

# ============================================================================
# EVALUATE MODEL
# ============================================================================

print(f"\n📈 Evaluating Model Performance...")

# Predictions
y_pred = rf_classifier.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {accuracy:.1%}")

# Cross-validation score (5-fold)
cv_scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')
print(f"🎯 Cross-Validation Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")

# Classification report
print(f"\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix
print(f"\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
crime_types = sorted(y.unique())
print(f"\n{'':15}", end='')
for crime in crime_types:
    print(f"{crime:15}", end='')
print()
for i, crime in enumerate(crime_types):
    print(f"{crime:15}", end='')
    for j in range(len(crime_types)):
        print(f"{cm[i,j]:15}", end='')
    print()

# Feature importance
print(f"\n🔍 Feature Importance (Top 5):")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['feature']:20} {row['importance']:.3f}")

# ============================================================================
# SAVE MODEL
# ============================================================================

print(f"\n💾 Saving trained model...")

# Save the classifier
joblib.dump(rf_classifier, 'crime_classifier_rf.pkl')
print(f"✓ Model saved: crime_classifier_rf.pkl")

# Save feature information
feature_info = {
    'feature_columns': feature_columns,
    'crime_types': list(y.unique())
}
joblib.dump(feature_info, 'feature_info.pkl')
print(f"✓ Feature info saved: feature_info.pkl")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✓ CRIME CLASSIFICATION TRAINING COMPLETE!")
print("=" * 80)
print(f"\n📊 Final Results:")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Test Accuracy: {accuracy:.1%}")
print(f"   Cross-Val Accuracy: {cv_scores.mean():.1%}")
print(f"   Crime types: {', '.join(crime_types)}")

print(f"\n📁 Output Files:")
print(f"   • crime_classifier_rf.pkl - Trained Random Forest model")
print(f"   • feature_info.pkl - Feature columns and crime types")

print(f"\n📋 Next Steps:")
print(f"   1. Integrate classifier into Streamlit dashboard")
print(f"   2. Test on new crime scene images")
print(f"   3. Fine-tune if needed")

if accuracy >= 0.85:
    print(f"\n🎉 SUCCESS! Achieved target accuracy of 85%+")
elif accuracy >= 0.80:
    print(f"\n✓ Good performance! Close to 85% target")
else:
    print(f"\n⚠ Below target. Consider:")
    print(f"   • Adding more training samples")
    print(f"   • Reviewing mislabeled images")
    print(f"   • Feature engineering")

print("=" * 80)
