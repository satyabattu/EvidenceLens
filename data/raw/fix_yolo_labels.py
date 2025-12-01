#!/usr/bin/env python3
"""
Fix YOLO label files by mapping filename patterns to correct class IDs
Run this in your /data/raw/ directory
"""

import os
import re
from pathlib import Path

# Class mapping based on filename patterns
CLASS_MAPPING = {
    'background_': 0,
    'blood_': 1,
    'bullet_hole_': 2,
    'cig_': 3,
    'Crimescene_tapes_': 4,
    'drink_can_': 5,
    'fingerprint_': 6,
    'footprint_': 7,
    'gloves_': 8,
    'gun_': 9,
    'knife_': 10,
    'shell_': 11,
    'shoeprint_': 12,
    'syringe_': 13
}

CLASS_NAMES = [
    'background',
    'blood',
    'bullet_hole',
    'cigarette',
    'crime_scene_tape',
    'drink_can',
    'fingerprint',
    'footprint',
    'gloves',
    'gun',
    'knife',
    'shell_casing',
    'shoeprint',
    'syringe'
]

def get_class_id_from_filename(filename):
    """Extract class ID from filename pattern"""
    for pattern, class_id in CLASS_MAPPING.items():
        if filename.startswith(pattern):
            return class_id
    
    # Handle edge cases
    print(f"WARNING: Unknown pattern for {filename}, defaulting to class 0")
    return 0

def fix_label_file(txt_path):
    """Fix class IDs in a single YOLO label file"""
    filename = os.path.basename(txt_path)
    correct_class_id = get_class_id_from_filename(filename)
    
    # Read original content
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Fix each line (replace first number with correct class ID)
    fixed_lines = []
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()
            # Replace class ID (first element) with correct one
            parts[0] = str(correct_class_id)
            fixed_lines.append(' '.join(parts) + '\n')
    
    # Write back
    with open(txt_path, 'w') as f:
        f.writelines(fixed_lines)
    
    return correct_class_id, len(fixed_lines)

def main():
    # Get current directory
    raw_dir = Path.cwd()
    
    print("=" * 60)
    print("YOLO LABEL FIX SCRIPT")
    print("=" * 60)
    print(f"Working directory: {raw_dir}")
    print()
    
    # Find all .txt files (excluding classes.txt if it exists)
    txt_files = [f for f in raw_dir.glob('*.txt') if f.name != 'classes.txt']
    
    print(f"Found {len(txt_files)} label files to fix")
    print()
    
    # Statistics
    class_counts = {i: 0 for i in range(14)}
    total_objects = 0
    
    # Process each file
    print("Processing files...")
    for i, txt_file in enumerate(txt_files, 1):
        class_id, num_objects = fix_label_file(txt_file)
        class_counts[class_id] += 1
        total_objects += num_objects
        
        if i % 100 == 0:
            print(f"  Processed {i}/{len(txt_files)} files...")
    
    print(f"✓ Processed {len(txt_files)}/{len(txt_files)} files")
    print()
    
    # Display statistics
    print("=" * 60)
    print("FIX COMPLETE - CLASS DISTRIBUTION:")
    print("=" * 60)
    for class_id, count in sorted(class_counts.items()):
        class_name = CLASS_NAMES[class_id]
        percentage = (count / len(txt_files) * 100) if txt_files else 0
        print(f"Class {class_id:2d} ({class_name:20s}): {count:3d} images ({percentage:5.1f}%)")
    
    print()
    print(f"Total images processed: {len(txt_files)}")
    print(f"Total objects labeled: {total_objects}")
    print()
    
    # Create classes.txt file
    classes_file = raw_dir / 'classes.txt'
    with open(classes_file, 'w') as f:
        for class_name in CLASS_NAMES:
            f.write(f"{class_name}\n")
    
    print(f"✓ Created {classes_file}")
    print()
    print("=" * 60)
    print("NEXT STEP: Run verify_fix.py to check random samples")
    print("=" * 60)

if __name__ == "__main__":
    main()