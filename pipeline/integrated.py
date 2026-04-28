# ============================================================
# Integrated Pipeline: End-to-End Analysis
# ============================================================

import torch
import json
from pathlib import Path
from PIL import Image
import yaml

from models.retrieval import CLIPRetriever
from models.vqa import VQAModel
from models.classification import ClassificationModel
from models.knowledge import KnowledgeBase
from models.xai import GradCAMExplainer

class AnalysisPipeline:
    """
    Integrated pipeline for Chinese cultural relics analysis.
    
    Pipeline:
    1. Image preprocessing
    2. CLIP embedding + FAISS retrieval (find similar images)
    3. Classification (material, type)
    4. VQA (answer questions)
    5. Knowledge lookup
    6. Grad-CAM explanation
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize pipeline.
        
        Args:
            config_path (str): Path to config file
        """
        self.config = self._load_config(config_path)
        self.device = self.config.get('device', 'cuda')
        
        print("🚀 Initializing VQA Cultural Relics Pipeline...")
        
        # Initialize components
        self.retriever = None
        self.vqa_model = None
        self.classifier = None
        self.knowledge_base = None
        self.explainer = None
    
    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to config file
            
        Returns:
            dict: Configuration
        """
        if not Path(config_path).exists():
            print(f"⚠️  Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def initialize_retriever(self, image_dir):
        """
        Initialize CLIP retriever and build FAISS index.
        
        Args:
            image_dir (str): Directory containing images
        """
        print("\n📊 Initializing CLIP Retriever...")
        
        model_config = self.config.get('models', {}).get('clip', {})
        self.retriever = CLIPRetriever(
            model_name=model_config.get('model_name', 'ViT-B/32'),
            device=self.device
        )
        
        # Build index
        self.retriever.build_index(
            image_dir,
            quantize=self.config.get('models', {}).get('faiss', {}).get('quantize', False)
        )
        print("✅ Retriever initialized")
    
    def initialize_vqa(self):
        """
        Initialize VQA model.
        """
        print("\n🧠 Initializing VQA Model...")
        
        model_config = self.config.get('models', {}).get('vqa', {})
        self.vqa_model = VQAModel(
            model_type=model_config.get('model_type', 'llava'),
            device=self.device
        )
        print("✅ VQA model initialized")
    
    def initialize_classifier(self):
        """
        Initialize classification model.
        """
        print("\n📊 Initializing Classifier...")
        
        class_config = self.config.get('models', {}).get('classification', {})
        self.classifier = ClassificationModel(
            num_classes=class_config.get('num_classes', 50),
            backbone=class_config.get('backbone', 'efficientnet_b0'),
            device=self.device
        )
        print("✅ Classifier initialized")
    
    def initialize_knowledge_base(self):
        """
        Initialize knowledge base.
        """
        print("\n📚 Initializing Knowledge Base...")
        
        kb_path = self.config.get('data', {}).get('knowledge_base', 'data/knowledge_base.json')
        self.knowledge_base = KnowledgeBase(kb_path)
        print("✅ Knowledge base initialized")
    
    def analyze(self, image_path, questions=None, top_k=5):
        """
        Analyze an image end-to-end.
        
        Args:
            image_path (str): Path to image
            questions (list): Questions to answer
            top_k (int): Number of similar images to retrieve
            
        Returns:
            dict: Complete analysis result
        """
        print(f"\n🔍 Analyzing: {image_path}")
        
        result = {
            "image_path": image_path,
            "retrieval": None,
            "classification": None,
            "vqa": None,
            "knowledge": None
        }
        
        # 1. Retrieval
        if self.retriever:
            print("  → Retrieving similar images...")
            similar = self.retriever.search(image_path, k=top_k)
            result["retrieval"] = [
                {"path": path, "distance": float(dist)}
                for path, dist in similar
            ]
            print(f"    ✓ Found {len(similar)} similar images")
        
        # 2. Classification
        if self.classifier:
            print("  → Classifying image...")
            pred = self.classifier.predict(image_path, return_probs=False)
            result["classification"] = pred
            print(f"    ✓ Class: {pred.get('class_name', 'Unknown')}")
        
        # 3. VQA
        if self.vqa_model:
            print("  → Answering questions...")
            if questions is None:
                questions = self.config.get('questions', {}).get('diverse', [])
            
            vqa_results = self.vqa_model.answer_multiple_questions(image_path, questions[:5])
            result["vqa"] = vqa_results
            print(f"    ✓ Answered {len(vqa_results)} questions")
        
        # 4. Knowledge
        if self.knowledge_base and result["classification"]:
            print("  → Looking up knowledge...")
            class_name = result["classification"].get("class_name", "")
            knowledge = self.knowledge_base.search(class_name, top_k=3)
            result["knowledge"] = knowledge
            print(f"    ✓ Found {len(knowledge)} knowledge items")
        
        print("\n✅ Analysis complete!")
        return result
    
    def save_result(self, result, output_dir="outputs"):
        """
        Save analysis result to JSON.
        
        Args:
            result (dict): Analysis result
            output_dir (str): Output directory
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        filename = Path(result["image_path"]).stem + "_analysis.json"
        output_path = Path(output_dir) / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Result saved to {output_path}")
