# ============================================================
# Image Classification: Material & Type Detection
# ============================================================

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class ClassificationModel:
    """
    Classify images by material (stone, ceramic, bronze, etc.)
    and type (statue, painting, vessel, etc.)
    """
    
    def __init__(self, num_classes=50, backbone="efficientnet_b0", device="cuda"):
        """
        Initialize classification model.
        
        Args:
            num_classes (int): Number of classes
            backbone (str): "efficientnet_b0", "vit_b_16", "resnet50"
            device (str): "cuda" or "cpu"
        """
        self.device = device
        self.num_classes = num_classes
        self.backbone = backbone
        
        print(f"Loading {backbone} classifier...")
        
        if backbone == "efficientnet_b0":
            self.model = models.efficientnet_b0(pretrained=True)
            self.model.classifier[-1] = nn.Linear(1280, num_classes)
        elif backbone == "vit_b_16":
            self.model = models.vit_b_16(pretrained=True)
            self.model.heads[-1] = nn.Linear(768, num_classes)
        elif backbone == "resnet50":
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, num_classes)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.class_names = None
        print(f"✅ {backbone} loaded with {num_classes} classes")
    
    def set_class_names(self, names):
        """
        Set class names for interpretation.
        
        Args:
            names (list): List of class names
        """
        self.class_names = names
    
    def predict(self, image_path, return_probs=False):
        """
        Predict class for an image.
        
        Args:
            image_path (str): Path to image
            return_probs (bool): Return probabilities for all classes
            
        Returns:
            dict: Prediction result
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(image_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = logits.argmax(dim=1).item()
                pred_prob = probs[0, pred_idx].item()
            
            result = {
                "class_idx": pred_idx,
                "confidence": pred_prob
            }
            
            if self.class_names:
                result["class_name"] = self.class_names[pred_idx]
            
            if return_probs:
                result["probabilities"] = probs[0].cpu().numpy()
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images.
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            list: List of predictions
        """
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Checkpoint loaded from {checkpoint_path}")
    
    def save_checkpoint(self, checkpoint_path):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path (str): Path to save checkpoint
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {'model_state_dict': self.model.state_dict()},
            checkpoint_path
        )
        print(f"✅ Checkpoint saved to {checkpoint_path}")
