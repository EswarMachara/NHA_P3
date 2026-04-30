import re

import cv2
import numpy as np
import pytesseract

from bbox_extractor import extract_bboxes
from config import (
    BBOX_MIN_AREA,
    PHASH_DISTANCE_THRESHOLD,
    PHASH_PATCH_SIZE,
    PHASH_STRIDE,
    get_logger,
)


logger = get_logger(__name__)


def assign_categories(
    page_data: dict,
    heatmaps: dict,
    statistical_results: dict,
    thresholds: dict
) -> list[dict]:
    """
    Assign document manipulation categories based on fused signals.
    """
    def get_threshold(category: str) -> float | None:
        if not thresholds:
            return None
        return thresholds.get(category) or thresholds.get(category.lower()) or thresholds.get(category.upper())

    def safe_max(map_data: np.ndarray) -> float:
        if map_data is None or map_data.size == 0:
            return 0.0
        return float(np.max(map_data))

    def phash_patch(patch: np.ndarray) -> int:
        patch_float = patch.astype(np.float32)
        resized = cv2.resize(patch_float, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(resized)
        dct_low = dct[:8, :8]
        flat = dct_low.flatten()
        if flat.size <= 1:
            return 0
        median = float(np.median(flat[1:]))
        bits = flat > median
        value = 0
        for bit in bits:
            value = (value << 1) | int(bool(bit))
        return value

    def has_duplicate_patches(gray: np.ndarray) -> bool:
        if gray is None:
            return False
        height, width = gray.shape[:2]
        if height < PHASH_PATCH_SIZE or width < PHASH_PATCH_SIZE:
            return False
        gray_uint8 = gray.astype(np.uint8, copy=False)
        hashes: list[int] = []
        for y in range(0, height - PHASH_PATCH_SIZE + 1, PHASH_STRIDE):
            for x in range(0, width - PHASH_PATCH_SIZE + 1, PHASH_STRIDE):
                patch = gray_uint8[y:y + PHASH_PATCH_SIZE, x:x + PHASH_PATCH_SIZE]
                patch_hash = phash_patch(patch)
                for prev in hashes:
                    if (patch_hash ^ prev).bit_count() < PHASH_DISTANCE_THRESHOLD:
                        return True
                hashes.append(patch_hash)
                if len(hashes) > 2000:
                    return False
        return False

    results: list[dict] = []
    counts: dict[str, int] = {}

    image_rgb = page_data.get("image_rgb") if page_data else None
    image_gray = page_data.get("image_gray") if page_data else None

    ela_map = heatmaps.get("ela") if heatmaps else None
    c2_map = heatmaps.get("c2") if heatmaps else None
    c9_map = heatmaps.get("c9") if heatmaps else None

    try:
        c8_info = (statistical_results or {}).get("c8", {})
        if c8_info.get("is_ai_generated"):
            results.append(
                {
                    "category": "C8",
                    "confidence": 1.0,
                    "bboxes": [],
                    "extra": {},
                }
            )
            counts["C8"] = 1
    except Exception as exc:
        logger.warning("Failed C8 decision: %s", exc)

    try:
        c7_regions = (statistical_results or {}).get("c7", [])
        if isinstance(c7_regions, list) and len(c7_regions) > 0:
            bboxes = []
            max_stretch = 0.0
            for region in c7_regions:
                if not isinstance(region, dict):
                    continue
                x = int(region.get("x", 0))
                y = int(region.get("y", 0))
                w = int(region.get("w", 0))
                h = int(region.get("h", 0))
                stretch = float(region.get("stretch_factor", 0.0))
                max_stretch = max(max_stretch, stretch)
                bboxes.append(
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "stretch_factor": stretch,
                    }
                )
            if bboxes:
                results.append(
                    {
                        "category": "C7",
                        "confidence": float(max_stretch),
                        "bboxes": bboxes,
                        "extra": {},
                    }
                )
                counts["C7"] = len(bboxes)
    except Exception as exc:
        logger.warning("Failed C7 decision: %s", exc)

    try:
        threshold = get_threshold("C9")
        if c9_map is not None and threshold is not None:
            c9_boxes = extract_bboxes(c9_map, threshold, BBOX_MIN_AREA)
            if c9_boxes:
                types: list[str] = []
                new_vals: list[str] = []
                old_vals: list[str] = []
                for bbox in c9_boxes:
                    text = ""
                    if image_rgb is not None:
                        x = max(int(bbox["x"]), 0)
                        y = max(int(bbox["y"]), 0)
                        w = max(int(bbox["w"]), 0)
                        h = max(int(bbox["h"]), 0)
                        crop = image_rgb[y:y + h, x:x + w]
                        if crop.size > 0:
                            try:
                                text = pytesseract.image_to_string(
                                    crop,
                                    config="--psm 6 --oem 1",
                                )
                            except Exception:
                                text = ""
                    text = text.strip()

                    date_pattern = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
                    amount_pattern = re.search(r"\b\d{3,}\b", text)

                    if date_pattern:
                        type_value = "date"
                        old_val = "Date of Birth"
                    elif amount_pattern:
                        type_value = "amount"
                        old_val = "Total Amount"
                    else:
                        type_value = "name"
                        old_val = "Patient Name"

                    types.append(type_value)
                    new_vals.append(text)
                    old_vals.append(old_val)

                results.append(
                    {
                        "category": "C9",
                        "confidence": safe_max(c9_map),
                        "bboxes": c9_boxes,
                        "extra": {
                            "types": types,
                            "new": new_vals,
                            "old": old_vals,
                        },
                    }
                )
                counts["C9"] = len(c9_boxes)
    except Exception as exc:
        logger.warning("Failed C9 decision: %s", exc)

    try:
        threshold = get_threshold("C3")
        if ela_map is not None and threshold is not None:
            c3_boxes = extract_bboxes(ela_map, threshold, BBOX_MIN_AREA)
            if c3_boxes:
                types: list[str] = []
                edge_map = None
                if image_gray is not None:
                    gray_uint8 = image_gray.astype(np.uint8, copy=False)
                    edge_map = cv2.Canny(gray_uint8, 50, 150)
                for bbox in c3_boxes:
                    w = float(bbox["w"])
                    h = float(bbox["h"])
                    ratio = w / h if h > 0 else 0.0
                    if 0.7 <= ratio <= 1.3:
                        types.append("stamp")
                        continue
                    density = 0.0
                    if edge_map is not None:
                        x = max(int(bbox["x"]), 0)
                        y = max(int(bbox["y"]), 0)
                        w_int = max(int(bbox["w"]), 0)
                        h_int = max(int(bbox["h"]), 0)
                        roi = edge_map[y:y + h_int, x:x + w_int]
                        density = float(roi.mean() / 255.0) if roi.size > 0 else 0.0
                    if density > 0.15:
                        types.append("signature")
                    else:
                        types.append("text")

                results.append(
                    {
                        "category": "C3",
                        "confidence": safe_max(ela_map),
                        "bboxes": c3_boxes,
                        "extra": {"types": types},
                    }
                )
                counts["C3"] = len(c3_boxes)
    except Exception as exc:
        logger.warning("Failed C3 decision: %s", exc)

    try:
        threshold = get_threshold("C1")
        if ela_map is not None and threshold is not None:
            c1_boxes = extract_bboxes(ela_map, threshold, BBOX_MIN_AREA)
            if c1_boxes and has_duplicate_patches(image_gray):
                results.append(
                    {
                        "category": "C1",
                        "confidence": safe_max(ela_map),
                        "bboxes": c1_boxes,
                        "extra": {},
                    }
                )
                counts["C1"] = len(c1_boxes)
    except Exception as exc:
        logger.warning("Failed C1 decision: %s", exc)

    try:
        threshold = get_threshold("C4")
        if ela_map is not None and threshold is not None:
            c4_boxes = extract_bboxes(ela_map, threshold, BBOX_MIN_AREA)
            if c4_boxes:
                results.append(
                    {
                        "category": "C4",
                        "confidence": safe_max(ela_map),
                        "bboxes": c4_boxes,
                        "extra": {},
                    }
                )
                counts["C4"] = len(c4_boxes)
    except Exception as exc:
        logger.warning("Failed C4 decision: %s", exc)

    try:
        c5_info = (statistical_results or {}).get("c5", {})
        if c5_info.get("is_suspicious"):
            band_transitions = c5_info.get("band_transitions") or []
            if band_transitions and (image_gray is not None or image_rgb is not None):
                if image_gray is not None:
                    height, width = image_gray.shape[:2]
                else:
                    height, width = image_rgb.shape[:2]
                split_y = int(band_transitions[0])
                split_y = min(max(split_y, 0), height)
                if split_y < height // 2:
                    bbox = {"x": 0, "y": 0, "w": int(width), "h": int(split_y)}
                    types = ["header"]
                else:
                    bbox = {"x": 0, "y": int(split_y), "w": int(width), "h": int(height - split_y)}
                    types = ["body"]

                results.append(
                    {
                        "category": "C5",
                        "confidence": float(c5_info.get("num_transitions", 0)),
                        "bboxes": [bbox],
                        "extra": {"types": types},
                    }
                )
                counts["C5"] = 1
    except Exception as exc:
        logger.warning("Failed C5 decision: %s", exc)

    try:
        threshold = get_threshold("C2")
        if c2_map is not None and threshold is not None:
            c2_boxes = extract_bboxes(c2_map, threshold, BBOX_MIN_AREA)
            if c2_boxes:
                results.append(
                    {
                        "category": "C2",
                        "confidence": safe_max(c2_map),
                        "bboxes": c2_boxes,
                        "extra": {},
                    }
                )
                counts["C2"] = len(c2_boxes)
    except Exception as exc:
        logger.warning("Failed C2 decision: %s", exc)

    if not results:
        results = [
            {
                "category": "C10",
                "confidence": 0.0,
                "bboxes": [],
                "extra": {},
            }
        ]
        counts["C10"] = 1

    results.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)

    logger.info("Detected categories: %s", ", ".join([r["category"] for r in results]))
    for category, count in counts.items():
        logger.debug("Category %s count: %d", category, count)

    return results
