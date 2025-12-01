"""
EvidenceLens - Feature Extraction for Crime Classification
Extracts evidence counts from images using trained YOLO models
"""

from ultralytics import YOLO
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

print("=" * 80)
print("EVIDENCELENS - EVIDENCE FEATURE EXTRACTION")
print("Extracting evidence counts for crime classification training")
print("=" * 80)

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

# Model paths - using your best trained models
MODEL_PATHS = [
    'runs/detect/evidencelens_production/weights/best.pt',  # Latest production model
    'runs/detect/evidencelens_v2/weights/best.pt',          # Backup model if available
]

# Image directory - where your labeled images are stored
# Using data/raw to process ALL labeled images (train + val = ~910 images)
IMAGE_DIR = 'data/raw'  # ALL labeled images
# Alternative: Use split datasets
# IMAGE_DIR = 'data/train/images'  # Training set only (~728 images)
# IMAGE_DIR = 'data/val/images'   # Validation set only (~182 images)

# Output CSV file
OUTPUT_CSV = 'evidence_features.csv'

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.25

# Class mapping (your 11 evidence classes)
CLASS_NAMES = {
    0: 'background',
    1: 'blood',
    2: 'bullet_hole',
    3: 'crime_scene_tape',
    4: 'drink_can',
    5: 'fingerprint',
    6: 'gloves',
    7: 'gun',
    8: 'knife',
    9: 'shell_casing',
    10: 'prints'
}

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model():
    """Load the best available trained model"""
    for model_path in MODEL_PATHS:
        try:
            model = YOLO(model_path)
            print(f"✓ Loaded model: {model_path}")
            return model
        except Exception as e:
            print(f"⚠ Could not load {model_path}: {e}")
            continue
    
    print("❌ ERROR: No trained models found!")
    print("Please check your model paths in the configuration section.")
    sys.exit(1)

# ============================================================================
# EXTRACT FEATURES FROM IMAGES
# ============================================================================

def extract_features_from_image(model, image_path, conf_threshold=0.25):
    """
    Run YOLO detection on an image and count evidence by class
    
    Returns:
        dict: Evidence counts for each class
    """
    # Initialize counts for all classes
    evidence_counts = {class_name: 0 for class_name in CLASS_NAMES.values()}
    evidence_counts['image_name'] = image_path.name
    
    try:
        # Run detection
        results = model.predict(str(image_path), conf=conf_threshold, verbose=False)
        
        # Count detections by class
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id in CLASS_NAMES:
                class_name = CLASS_NAMES[class_id]
                evidence_counts[class_name] += 1
        
        return evidence_counts
    
    except Exception as e:
        print(f"⚠ Error processing {image_path.name}: {e}")
        return evidence_counts

# ============================================================================
# MAIN EXTRACTION PROCESS
# ============================================================================

def main():
    # Load model
    model = load_model()
    
    # Get all images
    image_dir = Path(IMAGE_DIR)
    if not image_dir.exists():
        print(f"❌ ERROR: Image directory not found: {IMAGE_DIR}")
        sys.exit(1)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(image_dir.glob(ext)))
    
    if len(all_images) == 0:
        print(f"❌ ERROR: No images found in {IMAGE_DIR}")
        sys.exit(1)
    
    print(f"\n📸 Found {len(all_images)} images")
    print(f"📊 Extracting evidence features with confidence ≥ {CONFIDENCE_THRESHOLD}")
    print(f"💾 Output will be saved to: {OUTPUT_CSV}\n")
    
    # Extract features from all images
    results_list = []
    
    for image_path in tqdm(all_images, desc="Processing images"):
        evidence_counts = extract_features_from_image(
            model, 
            image_path, 
            conf_threshold=CONFIDENCE_THRESHOLD
        )
        results_list.append(evidence_counts)
    
    # Create DataFrame
    df = pd.DataFrame(results_list)
    
    # Reorder columns - image_name first, then all evidence classes
    columns_order = ['image_name'] + [name for name in CLASS_NAMES.values()]
    df = df[columns_order]
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("✓ FEATURE EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Summary Statistics:")
    print(f"   Total images processed: {len(df)}")
    print(f"   Output saved to: {OUTPUT_CSV}")
    print(f"\n🔍 Evidence Detection Summary:")
    
    # Show counts for each evidence type
    for class_name in CLASS_NAMES.values():
        if class_name != 'background':  # Skip background class
            count = (df[class_name] > 0).sum()
            total_detections = df[class_name].sum()
            print(f"   {class_name.replace('_', ' ').title()}: "
                  f"{count} images ({total_detections} total detections)")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Open {OUTPUT_CSV} to view the extracted features")
    print(f"   2. Create crime_labels.csv with crime types for each image")
    print(f"   3. Run the classification training script")
    print("=" * 80)

if __name__ == "__main__":
    main()
