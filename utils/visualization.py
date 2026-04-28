# ============================================================
# Visualization Utilities
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def plot_retrieval_results(query_image, results, titles=None, figsize=(15, 5)):
    """
    Visualize retrieval results.
    
    Args:
        query_image (str): Path to query image
        results (list): List of (image_path, distance) tuples
        titles (list): Optional titles
        figsize (tuple): Figure size
    """
    num_results = min(len(results) + 1, 6)  # Query + top 5
    
    fig, axes = plt.subplots(1, num_results, figsize=figsize)
    if num_results == 1:
        axes = [axes]
    
    # Query image
    query_img = mpimg.imread(query_image)
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image")
    axes[0].axis("off")
    
    # Results
    for idx, (result_path, distance) in enumerate(results[:num_results-1]):
        result_img = mpimg.imread(result_path)
        axes[idx+1].imshow(result_img)
        
        title = f"Rank {idx+1}\nDist: {distance:.3f}"
        if titles and idx < len(titles):
            title = f"{titles[idx]}\n{title}"
        
        axes[idx+1].set_title(title)
        axes[idx+1].axis("off")
    
    plt.tight_layout()
    return fig

def plot_classification_results(image, prediction, confidence):
    """
    Visualize classification result.
    
    Args:
        image (str): Path to image
        prediction (str): Predicted class
        confidence (float): Confidence score
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    img = mpimg.imread(image)
    ax.imshow(img)
    
    title = f"Prediction: {prediction}\nConfidence: {confidence:.2%}"
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    
    plt.tight_layout()
    return fig
