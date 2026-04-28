# ============================================================
# XAI: Explainable AI using Grad-CAM
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms

class GradCAMExplainer:
    """
    Generate Grad-CAM visualizations to explain model predictions.
    Highlight regions of interest that influenced the classification.
    """
    
    def __init__(self, model, target_layer, device="cuda"):
        """
        Initialize Grad-CAM explainer.
        
        Args:
            model: Pytorch model
            target_layer: Name of layer to hook (e.g., "layer4")
            device: "cuda" or "cpu"
        """
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register forward and backward hooks.
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        target = None
        for name, module in self.model.named_modules():
            if self.target_layer in name:
                target = module
                break
        
        if target is None:
            raise ValueError(f"Layer {self.target_layer} not found")
        
        target.register_forward_hook(forward_hook)
        target.register_backward_hook(backward_hook)
    
    def generate_cam(self, image_tensor, class_idx):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index
            
        Returns:
            np.ndarray: CAM heatmap (H, W)
        """
        # Forward pass
        self.model.eval()
        logits = self.model(image_tensor)
        
        # Backward pass
        self.model.zero_grad()
        target = logits[0, class_idx]
        target.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        weights = gradients.mean(axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def explain_image(self, image_path, class_idx, save_path=None):
        """
        Generate explanation for image classification.
        
        Args:
            image_path (str): Path to image
            class_idx (int): Target class index
            save_path (str): Path to save visualization
            
        Returns:
            np.ndarray: Visualization image
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        original_image = np.array(image.resize((224, 224)))
        
        # Generate CAM
        cam = self.generate_cam(image_tensor, class_idx)
        
        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (224, 224))
        
        # Create visualization
        cam_color = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_image, 0.6, cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB), 0.4, 0)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        return overlay
    
    def visualize_attention(self, heatmap, original_image=None, title="Attention"):
        """
        Visualize attention heatmap.
        
        Args:
            heatmap: Attention heatmap
            original_image: Original image for overlay
            title: Plot title
        """
        fig, axes = plt.subplots(1, 3 if original_image is not None else 1, figsize=(15, 5))
        
        if original_image is not None:
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            axes[1].imshow(heatmap, cmap="hot")
            axes[1].set_title("Attention Heatmap")
            axes[1].axis("off")
            
            overlay = 0.6 * original_image + 0.4 * cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis("off")
        else:
            axes.imshow(heatmap, cmap="hot")
            axes.set_title(title)
            axes.axis("off")
        
        plt.tight_layout()
        return fig
