import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from src.config import WEIGHTS_DIR, get_logger


logger = get_logger(__name__)


def load_catnet_model(device: str = "cpu"):
    """
    Load CAT-Net model from local weights file.
    """
    weights_path = os.path.join(WEIGHTS_DIR, "catnet.pth")
    if not os.path.exists(weights_path):
        logger.warning("CAT-Net weights not found: %s", weights_path)
        raise FileNotFoundError(weights_path)

    try:
        model = torch.jit.load(weights_path, map_location=device)
    except Exception:
        try:
            model = torch.load(weights_path, map_location=device)
        except Exception as exc:
            logger.warning("Failed to load CAT-Net weights from %s: %s", weights_path, exc)
            raise
        if isinstance(model, dict):
            if "model" in model:
                model = model["model"]
            elif "state_dict" in model:
                logger.warning("CAT-Net checkpoint contains state_dict only: %s", weights_path)
                raise ValueError("CAT-Net weights require a serialized model")

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    return model


def run_catnet(model, image_rgb: np.ndarray, is_jpeg: bool, device: str = "cpu") -> np.ndarray:
    """
    Run CAT-Net inference and return a heatmap.
    """
    try:
        if image_rgb is None:
            raise ValueError("image_rgb is None")

        height, width = image_rgb.shape[:2]
        if not is_jpeg:
            return np.zeros((height, width), dtype=np.float32)

        image_uint8 = image_rgb.astype(np.uint8, copy=False)
        tensor = transforms.ToTensor()(image_uint8)
        input_tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        if isinstance(output, (list, tuple)):
            output = output[0]
        elif isinstance(output, dict):
            output = next(iter(output.values()))

        if not torch.is_tensor(output):
            raise ValueError("CAT-Net output is not a tensor")

        output_tensor = output.squeeze()
        output_np = output_tensor.detach().float().cpu().numpy()
        if output_np.ndim == 3:
            output_np = output_np[0]
        if output_np.shape != (height, width):
            output_np = cv2.resize(output_np, (width, height), interpolation=cv2.INTER_LINEAR)

        heatmap = output_np.astype(np.float32)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        logger.debug("CAT-Net heatmap min/max: %.6f/%.6f", float(heatmap.min()), float(heatmap.max()))
        return heatmap
    except Exception as exc:
        logger.warning("Failed to run CAT-Net: %s", exc)
        try:
            height, width = image_rgb.shape[:2]
        except Exception:
            height, width = 0, 0
        return np.zeros((height, width), dtype=np.float32)
