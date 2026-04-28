# ============================================================
# CLIP + FAISS Retrieval System
# ============================================================

import torch
import clip
import faiss
import numpy as np
from PIL import Image
from pathlib import Path
import pickle
from tqdm import tqdm
import os

class CLIPRetriever:
    """
    CLIP + FAISS based image retrieval system.
    
    Encodes images to embeddings using CLIP and builds
    FAISS index for fast similarity search.
    """
    
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        """
        Initialize CLIP retriever.
        
        Args:
            model_name (str): CLIP model name (ViT-B/32, ViT-L/14, etc.)
            device (str): Device to use ("cuda" or "cpu")
        """
        self.device = device
        self.model_name = model_name
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
        self.embeddings = None
        self.index = None
        self.image_paths = []
        self.embedding_dim = 512 if "ViT-B" in model_name else 768
        
    def encode_image(self, image_path):
        """
        Encode single image to CLIP embedding.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.ndarray: Image embedding (512,) or (768,)
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy()[0]
        except Exception as e:
            print(f"Error encoding {image_path}: {e}")
            return None
    
    def build_index(self, image_dir, quantize=False):
        """
        Build FAISS index from all images in directory.
        
        Args:
            image_dir (str): Directory containing images
            quantize (bool): Use product quantization to reduce memory
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
        image_paths = [
            str(p) for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in image_extensions
        ]
        
        print(f"Found {len(image_paths)} images")
        
        embeddings = []
        self.image_paths = []
        
        for img_path in tqdm(image_paths, desc="Encoding images"):
            emb = self.encode_image(img_path)
            if emb is not None:
                embeddings.append(emb)
                self.image_paths.append(img_path)
        
        if not embeddings:
            raise ValueError("No valid images found!")
        
        self.embeddings = np.array(embeddings, dtype=np.float32)
        print(f"Embeddings shape: {self.embeddings.shape}")
        
        # Build FAISS index
        if quantize:
            # Product Quantization for smaller index
            nlist = min(100, len(embeddings) // 10)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFPQ(
                quantizer, self.embedding_dim, nlist, 8, 8
            )
            self.index.train(self.embeddings)
        else:
            # Flat index for exact search
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.index.add(self.embeddings)
        print(f"✅ FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query_path, k=5):
        """
        Search for similar images.
        
        Args:
            query_path (str): Path to query image
            k (int): Number of results to return
            
        Returns:
            list: List of (image_path, distance) tuples
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index() first.")
        
        query_emb = self.encode_image(query_path)
        if query_emb is None:
            return []
        
        query_emb = query_emb.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_emb, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(dist)))
        
        return results
    
    def search_batch(self, query_paths, k=5):
        """
        Batch search for multiple query images.
        
        Args:
            query_paths (list): List of query image paths
            k (int): Number of results per query
            
        Returns:
            dict: Mapping from query path to results
        """
        results = {}
        for qp in tqdm(query_paths, desc="Batch search"):
            results[qp] = self.search(qp, k)
        return results
    
    def save_index(self, output_path):
        """
        Save FAISS index to disk.
        
        Args:
            output_path (str): Path to save index
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "index": self.index,
                "embeddings": self.embeddings,
                "image_paths": self.image_paths,
                "model_name": self.model_name
            }, f)
        print(f"✅ Index saved to {output_path}")
    
    def load_index(self, input_path):
        """
        Load FAISS index from disk.
        
        Args:
            input_path (str): Path to saved index
        """
        with open(input_path, "rb") as f:
            data = pickle.load(f)
            self.index = data["index"]
            self.embeddings = data["embeddings"]
            self.image_paths = data["image_paths"]
            self.model_name = data["model_name"]
        print(f"✅ Index loaded from {input_path}")
