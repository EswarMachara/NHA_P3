import os

import cv2
import numpy as np
import torch

from src.config import WEIGHTS_DIR, get_logger
from src.mvss.mvssnet import get_mvss


logger = get_logger(__name__)


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    if not all(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    return {key[len(prefix):]: value for key, value in state_dict.items()}


def load_mvss_model(device: str):
    """
    Load MVSS-Net model from local weights file.
    """
    candidates = [
        os.path.join(WEIGHTS_DIR, "mvssnet_casia.pt"),
        os.path.join(WEIGHTS_DIR, "mvssnet_defacto.pt"),
        os.path.join(WEIGHTS_DIR, "mvssnet.pth"),
        os.path.join(WEIGHTS_DIR, "mvssnet.pt"),
    ]
    weights_path = next((path for path in candidates if os.path.exists(path)), None)
    if not weights_path:
        logger.warning("MVSS-Net weights not found in %s", WEIGHTS_DIR)
        raise FileNotFoundError("MVSS-Net weights not found")

    model = get_mvss(
        backbone="resnet50",
        pretrained_base=False,
        nclass=1,
        sobel=True,
        n_input=3,
        constrain=True,
    )

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("MVSS-Net checkpoint does not contain a state_dict")

    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "model.")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        logger.warning("MVSS-Net strict load failed (%s). Retrying with strict=False.", exc)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    return model


def _prepare_mvss_input(image_rgb: np.ndarray) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    tensor = torch.from_numpy(image.transpose(2, 0, 1))
    return tensor.unsqueeze(0)


def run_mvss(model, image_rgb: np.ndarray, device: str) -> tuple[np.ndarray, dict]:
    """
    Run MVSS-Net inference and return an anomaly heatmap.
    """
    try:
        if image_rgb is None:
            raise ValueError("image_rgb is None")

        height, width = image_rgb.shape[:2]
        input_tensor = _prepare_mvss_input(image_rgb).to(device)

        with torch.no_grad():
            edge_map, seg_map = model(input_tensor)

        if isinstance(seg_map, (list, tuple)):
            seg_map = seg_map[0]
        elif isinstance(seg_map, dict):
            seg_map = next(iter(seg_map.values()))

        if not torch.is_tensor(seg_map):
            raise ValueError("MVSS-Net output is not a tensor")

        output_tensor = torch.sigmoid(seg_map).squeeze()
        output_np = output_tensor.detach().float().cpu().numpy()
        if output_np.ndim == 3:
            output_np = output_np[0]
        if output_np.shape != (height, width):
            output_np = cv2.resize(output_np, (width, height), interpolation=cv2.INTER_LINEAR)

        heatmap = output_np.astype(np.float32)
        heatmap = np.clip(heatmap, 0.0, 1.0)

        logger.debug("MVSS heatmap min/max: %.6f/%.6f", float(heatmap.min()), float(heatmap.max()))
        return heatmap, {}
    except Exception as exc:
        logger.warning("Failed to run MVSS-Net: %s", exc)
        try:
            height, width = image_rgb.shape[:2]
        except Exception:
            height, width = 0, 0
        return np.zeros((height, width), dtype=np.float32), {}
