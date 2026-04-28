#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Start Demo - VQA Cultural Relics System
Chạy file này để test toàn bộ pipeline
Usage: python notebooks/00_quick_start.py
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*60)
print("🚀 VQA CULTURAL RELICS - QUICK START DEMO")
print("="*60)

# ============================================================
# STEP 1: Setup & Imports
# ============================================================
print("\n[STEP 1] 📦 Importing libraries...")
try:
    import torch
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    print("✅ Core libraries imported")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\n💡 Please install requirements first:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# ============================================================
# STEP 2: Check Device
# ============================================================
print("\n[STEP 2] 🖥️  Checking device...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")
if device == "cpu":
    print("⚠️  GPU not available. Using CPU (slower)")

# ============================================================
# STEP 3: Create Sample Data
# ============================================================
print("\n[STEP 3] 🖼️  Creating sample images for testing...")

# Create data directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Create 3 sample images (256x256 random colored)
def create_sample_image(filename, color_seed=0):
    """Create a simple colored image for testing"""
    np.random.seed(color_seed)
    img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    img.save(filename)
    return filename

sample_images = []
for i in range(3):
    img_path = f"data/raw/sample_image_{i}.jpg"
    create_sample_image(img_path, color_seed=i)
    sample_images.append(img_path)
    print(f"  ✓ Created: {img_path}")

print(f"✅ Created {len(sample_images)} sample images")

# ============================================================
# STEP 4: Initialize Models (with error handling)
# ============================================================
print("\n[STEP 4] 🤖 Initializing models...\n")

# 4.1 CLIP Retriever
print("  [4.1] Initializing CLIP Retriever...")
retriever = None
try:
    from models.retrieval import CLIPRetriever
    retriever = CLIPRetriever(model_name="ViT-B/32", device=device)
    print("       ✅ CLIP Retriever ready\n")
except Exception as e:
    print(f"       ❌ CLIP Error: {e}\n")

# 4.2 VQA Model - Skip for now (BLIP-2 has issues)
print("  [4.2] Initializing VQA Model...")
vqa = None
print("       ⚠️  Skipping VQA (transformer version issue)")
print("       💡 Use LLaVA or BLIP later\n")

# 4.3 Classification Model
print("  [4.3] Initializing Classification Model...")
classifier = None
try:
    from models.classification import ClassificationModel
    classifier = ClassificationModel(num_classes=10, backbone="efficientnet_b0", device=device)
    classifier.set_class_names([
        "bronze", "stone", "ceramic", "statue", "vessel",
        "painting", "jade", "lacquer", "wood", "other"
    ])
    print("       ✅ Classifier ready\n")
except Exception as e:
    print(f"       ❌ Classifier Error: {e}\n")

# 4.4 Knowledge Base
print("  [4.4] Initializing Knowledge Base...")
kb = None
try:
    from models.knowledge import KnowledgeBase
    kb = KnowledgeBase()
    print("       ✅ Knowledge Base ready\n")
except Exception as e:
    print(f"       ❌ Knowledge Base Error: {e}\n")

# ============================================================
# STEP 5: Test Individual Components
# ============================================================
print("\n[STEP 5] 🧪 Testing individual components...\n")

test_image = sample_images[0]

# 5.1 Test Classification
if classifier:
    print("  [5.1] Testing Classification...")
    try:
        pred = classifier.predict(test_image, return_probs=False)
        print(f"       ✅ Prediction: {pred.get('class_name', 'Unknown')}")
        print(f"          Confidence: {pred.get('confidence', 0):.2%}\n")
    except Exception as e:
        print(f"       ❌ Error: {e}\n")

# 5.2 Test VQA - SKIP
print("  [5.2] Testing VQA...")
print("       ⚠️  Skipping VQA test\n")

# 5.3 Test Knowledge Base
if kb:
    print("  [5.3] Testing Knowledge Base...")
    try:
        material_info = kb.get_material_info("bronze")
        print(f"       ✅ Bronze info: {material_info.get('description', 'N/A')}\n")
    except Exception as e:
        print(f"       ❌ Error: {e}\n")

# 5.4 Test Retriever (build index first)
if retriever:
    print("  [5.4] Testing Retriever...")
    try:
        print("       Building FAISS index...")
        retriever.build_index("data/raw")
        print("       Searching similar images...")
        results = retriever.search(test_image, k=2)
        print(f"       ✅ Found {len(results)} similar images")
        for path, dist in results:
            print(f"          - {Path(path).name} (distance: {dist:.4f})\n")
    except Exception as e:
        print(f"       ❌ Error: {e}\n")

# ============================================================
# STEP 6: Simplified Pipeline (without VQA)
# ============================================================
print("\n[STEP 6] 🚀 Running Simplified Pipeline...\n")

try:
    # Initialize pipeline components manually
    print("  Initializing pipeline components...")
    
    # Preprocess images
    print("  Preprocessing images...")
    from utils.preprocessing import preprocess_images
    preprocess_images("data/raw", "data/processed")
    
    # Display results
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS")
    print("="*60)

    print("\n[Classification]")
    if classifier:
        pred = classifier.predict(test_image, return_probs=False)
        print(f"  Class: {pred.get('class_name', 'Unknown')}")
        print(f"  Confidence: {pred.get('confidence', 0):.2%}")

    print("\n[Retrieval - Similar Images]")
    if retriever:
        results = retriever.search(test_image, k=2)
        for i, (path, dist) in enumerate(results, 1):
            print(f"  {i}. {Path(path).name} (distance: {dist:.4f})")

    print("\n[Knowledge Base]")
    if kb:
        materials = kb.data.get("materials", {})
        print(f"  Available materials: {len(materials)}")
        for mat_key in list(materials.keys())[:3]:
            print(f"    - {mat_key}")

    # Save result
    os.makedirs("outputs", exist_ok=True)
    result = {
        "image": test_image,
        "classification": classifier.predict(test_image) if classifier else None,
        "retrieval": [(p, float(d)) for p, d in retriever.search(test_image, k=2)] if retriever else None,
    }
    
    import json
    with open("outputs/demo_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)

except Exception as e:
    print(f"❌ Pipeline Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# STEP 7: Next Steps
# ============================================================
print("\n" + "="*60)
print("📚 NEXT STEPS")
print("="*60)

next_steps = """
✅ What Works:
  ✓ CLIP Retriever + FAISS search
  ✓ Classification (EfficientNet)
  ✓ Knowledge Base
  ✓ Preprocessing & visualization

⚠️  Known Issues:
  - BLIP-2 has JSON parsing issues (transformer version mismatch)
  - LLaVA requires separate installation

💡 Solutions:
  1. For VQA, install LLaVA separately:
     pip install git+https://github.com/haotian-liu/LLaVA.git

  2. Or downgrade transformers:
     pip install transformers==4.30.0

  3. Use simpler BLIP model:
     from models.vqa import VQAModel
     vqa = VQAModel(model_type="blip", device=device)

🎯 Next:
  1. Fix VQA model (choose one above)
  2. Run on real dataset
  3. Fine-tune classifier on your data
  4. Expand knowledge base
  5. Add prompt engineering for better answers

Good luck! 🚀
"""

print(next_steps)
print("\n" + "="*60 + "\n")
