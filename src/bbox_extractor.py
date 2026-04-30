import cv2
import numpy as np

from config import (
    BBOX_MIN_AREA,
    BBOX_MORPH_CLOSE_KERNEL,
    BBOX_MORPH_OPEN_KERNEL,
    get_logger,
)


logger = get_logger(__name__)


def extract_bboxes(heatmap: np.ndarray,
                   threshold: float,
                   min_area: int) -> list[dict]:
    """
    Convert a heatmap into bounding boxes via thresholding and morphology.
    """
    try:
        if heatmap is None:
            return []
        if heatmap.size == 0:
            return []
        if float(np.max(heatmap)) == 0.0:
            return []

        if min_area is None:
            min_area = BBOX_MIN_AREA

        binary = (heatmap >= threshold).astype(np.uint8) * 255
        binary = np.ascontiguousarray(binary)

        k_close = BBOX_MORPH_CLOSE_KERNEL | 1
        k_open = BBOX_MORPH_OPEN_KERNEL | 1

        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (k_close, k_close),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (k_open, k_open),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        boxes: list[dict] = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area < min_area:
                continue

            boxes.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                }
            )

        logger.debug(
            "BBoxes: components=%d boxes=%d",
            max(num_labels - 1, 0),
            len(boxes),
        )
        return boxes
    except Exception as exc:
        logger.warning("Failed to extract bounding boxes: %s", exc)
        return []
