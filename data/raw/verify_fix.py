#!/usr/bin/env python3
"""
Verify that the label fix worked correctly
Shows random samples from each class
"""

import os
import random
from pathlib import Path
from collections import defaultdict

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

def verify_labels():
    """Verify label files are correctly formatted"""
    raw_dir = Path.cwd()
    txt_files = [f for f in raw_dir.glob('*.txt') if f.name != 'classes.txt']
    
    print("=" * 70)
    print("YOLO LABEL VERIFICATION")
    print("=" * 70)
    print()
    
    # Group files by class
    files_by_class = defaultdict(list)
    errors = []
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            continue
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{txt_file.name}:{line_num} - Expected 5 values, got {len(parts)}")
                continue
            
            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                
                # Validate ranges
                if not (0 <= class_id <= 13):
                    errors.append(f"{txt_file.name}:{line_num} - Invalid class ID: {class_id}")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    errors.append(f"{txt_file.name}:{line_num} - Coordinates out of range [0,1]")
                
                files_by_class[class_id].append(txt_file.name)
            
            except ValueError as e:
                errors.append(f"{txt_file.name}:{line_num} - Parse error: {e}")
    
    # Show errors if any
    if errors:
        print("⚠️  ERRORS FOUND:")
        print("-" * 70)
        for error in errors[:10]:  # Show first 10
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        print()
    else:
        print("✓ No format errors found!")
        print()
    
    # Show class distribution with samples
    print("CLASS DISTRIBUTION & SAMPLES:")
    print("-" * 70)
    
    for class_id in range(14):
        class_name = CLASS_NAMES[class_id]
        file_list = list(set(files_by_class[class_id]))  # Unique files
        count = len(file_list)
        
        # Show random samples
        samples = random.sample(file_list, min(3, count)) if count > 0 else []
        sample_str = ", ".join(samples) if samples else "No files"
        
        print(f"Class {class_id:2d} | {class_name:20s} | {count:3d} files")
        print(f"         Samples: {sample_str}")
        print()
    
    print("=" * 70)
    print("VERIFICATION SUMMARY:")
    print("=" * 70)
    print(f"Total label files checked: {len(txt_files)}")
    print(f"Format errors found: {len(errors)}")
    print(f"Classes represented: {len([c for c in files_by_class if files_by_class[c]])}/14")
    print()
    
    if not errors:
        print("✓ ALL LABELS LOOK GOOD!")
        print()
        print("=" * 70)
        print("NEXT STEP: Ready for dataset split and training!")
        print("=" * 70)
    else:
        print("⚠️  FIX ERRORS BEFORE PROCEEDING")
    
    print()

if __name__ == "__main__":
    verify_labels()