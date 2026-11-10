"""
Image preprocessing and inference utility functions.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
from src.config import IMG_SIZE, CLASS_NAMES, SPECIES_METADATA


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = IMG_SIZE,
    rescale: bool = True
) -> np.ndarray:
    """
    Preprocess PIL Image for model inference.
    
    Args:
        image: PIL Image object
        target_size: Target tuple (width, height), default (224, 224)
        rescale: Whether to divide pixel values by 255.0
        
    Returns:
        np.ndarray with shape (1, height, width, 3)
    """
    # Ensure 3-channel RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Resize with high quality bicubic/bilinear filter
    img_resized = image.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    
    if rescale:
        img_array = img_array / 255.0
        
    # Add batch dimension: (1, H, W, C)
    return np.expand_dims(img_array, axis=0)


def predict_top_k(
    model: Any,
    image_array: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Runs model inference and returns the top-k predicted classes with metadata.
    
    Returns:
        List of dicts: [
            {
                "rank": 1,
                "class_name": "Aedes_aegypti",
                "probability": 0.892,
                "confidence_percent": "89.2%",
                "metadata": {...}
            },
            ...
        ]
    """
    raw_preds = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(raw_preds)[::-1][:k]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
        prob = float(raw_preds[idx])
        metadata = SPECIES_METADATA.get(class_name, {
            "common_name": class_name.replace("_", " "),
            "category": "Unknown",
            "vector_status": "Unknown",
            "pathogens": [],
            "habitat": "N/A",
            "risk_level": "Unknown"
        })
        
        results.append({
            "rank": rank,
            "class_name": class_name,
            "probability": prob,
            "confidence_percent": f"{prob * 100:.1f}%",
            "metadata": metadata
        })
        
    return results
