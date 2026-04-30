import os
from types import SimpleNamespace
from typing import Optional

import cv2
import jpegio
import numpy as np
import torch
import yaml

from src.catnet.network_CAT import get_seg_model
from src.config import WEIGHTS_DIR, get_logger


logger = get_logger(__name__)


def _load_catnet_config():
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "catnet", "config", "CAT_full.yaml")
    )
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    model_cfg = data.get("MODEL", {})
    extra = model_cfg.get("EXTRA", {})
    for stage_key in ("STAGE1", "STAGE2", "STAGE3", "STAGE4", "DC_STAGE3", "DC_STAGE4", "STAGE5"):
        stage = extra.get(stage_key)
        if isinstance(stage, dict) and "NUM_RANCHES" in stage:
            stage["NUM_BRANCHES"] = stage.pop("NUM_RANCHES")

    cfg = SimpleNamespace()
    cfg.MODEL = SimpleNamespace()
    cfg.MODEL.EXTRA = extra
    cfg.MODEL.PRETRAINED_RGB = ""
    cfg.MODEL.PRETRAINED_DCT = ""
    cfg.DATASET = SimpleNamespace()
    cfg.DATASET.NUM_CLASSES = int(data.get("DATASET", {}).get("NUM_CLASSES", 2))
    return cfg


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    if not all(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    return {key[len(prefix):]: value for key, value in state_dict.items()}


def load_catnet_model(device: str = "cpu"):
    """
    Load CAT-Net model and weights from local checkpoint.
    """
    candidates = [
        os.path.join(WEIGHTS_DIR, "CAT_full_v2.pth.tar"),
        os.path.join(WEIGHTS_DIR, "catnet.pth"),
        os.path.join(WEIGHTS_DIR, "CAT_full_v1.pth.tar"),
    ]
    weights_path = next((path for path in candidates if os.path.exists(path)), None)
    if not weights_path:
        logger.warning("CAT-Net weights not found in %s", WEIGHTS_DIR)
        raise FileNotFoundError("CAT-Net weights not found")

    cfg = _load_catnet_config()
    model = get_seg_model(cfg)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("CAT-Net checkpoint does not contain a state_dict")

    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "model.")

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model


def _get_jpeg_info(im_path: str, dct_channels: int = 1):
    jpeg = jpegio.read(str(im_path))
    num_channels = dct_channels

    ci = jpeg.comp_info
    need_scale = [[ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)]
    if num_channels == 3:
        if ci[0].v_samp_factor == ci[1].v_samp_factor == ci[2].v_samp_factor:
            need_scale[0][0] = need_scale[1][0] = need_scale[2][0] = 2
        if ci[0].h_samp_factor == ci[1].h_samp_factor == ci[2].h_samp_factor:
            need_scale[0][1] = need_scale[1][1] = need_scale[2][1] = 2
    else:
        need_scale[0][0] = 2
        need_scale[0][1] = 2

    dct_coef: list[np.ndarray] = []
    for i in range(num_channels):
        rows, cols = jpeg.coef_arrays[i].shape
        coef_view = jpeg.coef_arrays[i].reshape(rows // 8, 8, cols // 8, 8).transpose(0, 2, 1, 3)

        if need_scale[i][0] == 1 and need_scale[i][1] == 1:
            out_arr = np.zeros((rows * 2, cols * 2), dtype=np.float32)
            out_view = out_arr.reshape(rows * 2 // 8, 8, cols * 2 // 8, 8).transpose(0, 2, 1, 3)
            out_view[::2, ::2, :, :] = coef_view
            out_view[1::2, ::2, :, :] = coef_view
            out_view[::2, 1::2, :, :] = coef_view
            out_view[1::2, 1::2, :, :] = coef_view
        elif need_scale[i][0] == 1 and need_scale[i][1] == 2:
            out_arr = np.zeros((rows * 2, cols), dtype=np.float32)
            out_view = out_arr.reshape(rows * 2 // 8, 8, cols // 8, 8).transpose(0, 2, 1, 3)
            out_view[::2, :, :, :] = coef_view
            out_view[1::2, :, :, :] = coef_view
        elif need_scale[i][0] == 2 and need_scale[i][1] == 1:
            out_arr = np.zeros((rows, cols * 2), dtype=np.float32)
            out_view = out_arr.reshape(rows // 8, 8, cols * 2 // 8, 8).transpose(0, 2, 1, 3)
            out_view[:, ::2, :, :] = coef_view
            out_view[:, 1::2, :, :] = coef_view
        elif need_scale[i][0] == 2 and need_scale[i][1] == 2:
            out_arr = np.zeros((rows, cols), dtype=np.float32)
            out_view = out_arr.reshape(rows // 8, 8, cols // 8, 8).transpose(0, 2, 1, 3)
            out_view[:, :, :, :] = coef_view
        else:
            raise KeyError("Invalid DCT scaling factors")

        dct_coef.append(out_arr)

    qtables = [jpeg.quant_tables[ci[i].quant_tbl_no].astype(np.float32) for i in range(num_channels)]
    return dct_coef, qtables


def _build_catnet_inputs(image_rgb: np.ndarray, jpeg_path: str):
    dct_coef, qtables = _get_jpeg_info(jpeg_path, dct_channels=1)
    img_rgb = image_rgb.astype(np.float32, copy=False)

    height, width = img_rgb.shape[:2]
    crop_h = ((height + 7) // 8) * 8
    crop_w = ((width + 7) // 8) * 8

    if crop_h != height or crop_w != width:
        temp = np.full((crop_h, crop_w, 3), 127.5, dtype=np.float32)
        temp[:height, :width, :] = img_rgb
        img_rgb = temp

    dct_map = dct_coef[0].astype(np.float32, copy=False)
    if dct_map.shape[0] != crop_h or dct_map.shape[1] != crop_w:
        temp = np.zeros((crop_h, crop_w), dtype=np.float32)
        copy_h = min(dct_map.shape[0], crop_h)
        copy_w = min(dct_map.shape[1], crop_w)
        temp[:copy_h, :copy_w] = dct_map[:copy_h, :copy_w]
        dct_map = temp

    t_rgb = (torch.tensor(img_rgb.transpose(2, 0, 1), dtype=torch.float32) - 127.5) / 127.5
    t_dct_coef = torch.tensor(dct_map, dtype=torch.float32).unsqueeze(0)

    t_val = 20
    t_dct_vol = torch.zeros((t_val + 1, t_dct_coef.shape[1], t_dct_coef.shape[2]), dtype=torch.float32)
    t_dct_vol[0] += (t_dct_coef == 0).float().squeeze(0)
    for i in range(1, t_val):
        t_dct_vol[i] += (t_dct_coef == i).float().squeeze(0)
        t_dct_vol[i] += (t_dct_coef == -i).float().squeeze(0)
    t_dct_vol[t_val] += (t_dct_coef >= t_val).float().squeeze(0)
    t_dct_vol[t_val] += (t_dct_coef <= -t_val).float().squeeze(0)

    tensor = torch.cat([t_rgb, t_dct_vol], dim=0)
    qtable = torch.tensor(qtables[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor, qtable, (height, width)


def run_catnet(
    model,
    image_rgb: np.ndarray,
    is_jpeg: bool,
    file_path: Optional[str] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run CAT-Net inference and return a heatmap.
    """
    try:
        if image_rgb is None:
            raise ValueError("image_rgb is None")

        height, width = image_rgb.shape[:2]
        if not is_jpeg:
            return np.zeros((height, width), dtype=np.float32)

        if not file_path:
            raise ValueError("file_path is required for CAT-Net JPEG processing")

        input_tensor, qtable, (orig_h, orig_w) = _build_catnet_inputs(image_rgb, file_path)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        qtable = qtable.to(device)

        with torch.no_grad():
            output = model(input_tensor, qtable)

        if isinstance(output, (list, tuple)):
            output = output[0]
        elif isinstance(output, dict):
            output = next(iter(output.values()))

        if not torch.is_tensor(output):
            raise ValueError("CAT-Net output is not a tensor")

        output_tensor = output.squeeze()
        output_np = output_tensor.detach().float().cpu().numpy()
        if output_np.ndim == 3:
            output_np = output_np[1] if output_np.shape[0] > 1 else output_np[0]
        if output_np.shape != (orig_h, orig_w):
            output_np = cv2.resize(output_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

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
