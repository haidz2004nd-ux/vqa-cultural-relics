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

# 4.2 VQA Model
print("  [4.2] Initializing VQA Model...")
vqa = None
try:
    from models.vqa import VQAModel
    # Try LLaVA first, fallback to BLIP-2
    try:
        vqa = VQAModel(model_type="llava", device=device)
        print("       ✅ LLaVA VQA ready\n")
    except Exception as llava_err:
        print("       ⚠️  LLaVA failed, trying BLIP-2...")
        try:
            vqa = VQAModel(model_type="blip2", device=device)
            print("       ✅ BLIP-2 VQA ready\n")
        except Exception as blip2_err:
            print(f"       ⚠️  BLIP-2 also failed: {blip2_err}\n")
except Exception as e:
    print(f"       ❌ VQA Error: {e}\n")

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

# 5.2 Test VQA
if vqa:
    print("  [5.2] Testing VQA...")
    try:
        test_questions = [
            "What is this?",
            "What material is this?"
        ]
        answers = vqa.answer_multiple_questions(test_image, test_questions)
        for q, a in answers.items():
            print(f"       Q: {q}")
            print(f"       A: {a}\n")
    except Exception as e:
        print(f"       ❌ Error: {e}\n")

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
# STEP 6: Full Pipeline
# ============================================================
print("\n[STEP 6] 🚀 Running Full Pipeline...\n")

try:
    from pipeline.integrated import AnalysisPipeline

    # Initialize pipeline
    print("  Initializing pipeline...")
    pipeline = AnalysisPipeline("config.yaml")

    # Copy processed images
    print("  Preprocessing images...")
    from utils.preprocessing import preprocess_images
    preprocess_images("data/raw", "data/processed")

    # Initialize components
    print("  Initializing retriever...")
    if retriever:
        pipeline.retriever = retriever

    print("  Initializing VQA...")
    if vqa:
        pipeline.vqa_model = vqa

    print("  Initializing classifier...")
    if classifier:
        pipeline.classifier = classifier

    print("  Initializing knowledge base...")
    if kb:
        pipeline.knowledge_base = kb

    # Run analysis
    print(f"\n  Analyzing: {test_image}\n")

    test_questions = [
        "What is this object?",
        "What is the dominant color?",
        "What material might this be?"
    ]

    result = pipeline.analyze(test_image, questions=test_questions, top_k=2)

    # Display results
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS")
    print("="*60)

    print("\n[Classification]")
    if result["classification"]:
        print(f"  Class: {result['classification'].get('class_name', 'Unknown')}")
        print(f"  Confidence: {result['classification'].get('confidence', 0):.2%}")

    print("\n[Retrieval - Similar Images]")
    if result["retrieval"]:
        for i, item in enumerate(result["retrieval"], 1):
            print(f"  {i}. {Path(item['path']).name} (distance: {item['distance']:.4f})")

    print("\n[VQA - Answers]")
    if result["vqa"]:
        for q, a in result["vqa"].items():
            print(f"  Q: {q}")
            print(f"  A: {a}\n")

    print("\n[Knowledge]")
    if result["knowledge"]:
        print(f"  Found {len(result['knowledge'])} knowledge items")
        for item in result["knowledge"]:
            print(f"    - {item.get('key', 'Unknown')}")

    # Save result
    os.makedirs("outputs", exist_ok=True)
    pipeline.save_result(result)

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
1. 📁 Prepare your data:
   - Copy images to data/raw/
   - Create data/web_labels.csv (if needed)

2. 🔧 Configure pipeline:
   - Edit config.yaml for your needs
   - Adjust model types, batch sizes, etc.

3. 📊 Run on real data:
   from pipeline.integrated import AnalysisPipeline
   pipeline = AnalysisPipeline("config.yaml")
   result = pipeline.analyze("your_image.jpg")

4. 📓 Check notebooks:
   - notebooks/01_preprocessing.ipynb
   - notebooks/02_retrieval_setup.ipynb
   - notebooks/03_vqa_inference.ipynb

5. 🎯 Fine-tune for your domain:
   - Train classifier on your data
   - Expand knowledge base
   - Customize VQA prompts

Good luck! 🚀
"""

print(next_steps)
print("\n" + "="*60 + "\n")
