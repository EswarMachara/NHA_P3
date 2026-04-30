import io

import cv2
import numpy as np
import pytesseract
from PIL import Image
from scipy.fft import dctn
from scipy.stats import kurtosis
from skimage.measure import shannon_entropy

from src.config import (
    C7_IQR_MULTIPLIER,
    C7_MIN_STRETCH_FACTOR,
    C8_ENTROPY_BG_THRESHOLD,
    C8_SIGMA_LAP_THRESHOLD,
    get_logger,
)


logger = get_logger(__name__)


def compute_ela(image_rgb: np.ndarray,
                quality_high: int = 95,
                quality_low: int = 75) -> np.ndarray:
    """
    Compute Error Level Analysis (ELA) map for an RGB image.
    """
    try:
        if image_rgb is None:
            raise ValueError("image_rgb is None")

        height, width = image_rgb.shape[:2]
        image_uint8 = image_rgb.astype(np.uint8, copy=False)

        pil_img = Image.fromarray(image_uint8)

        buf_high = io.BytesIO()
        pil_img.save(buf_high, format="JPEG", quality=quality_high)
        buf_high.seek(0)
        # img_high = np.array(Image.open(buf_high))
        img_high = np.array(Image.open(buf_high).convert("RGB"))

        buf_low = io.BytesIO()
        Image.fromarray(img_high).save(buf_low, format="JPEG", quality=quality_low)
        buf_low.seek(0)
        # img_low = np.array(Image.open(buf_low))
        img_low = np.array(Image.open(buf_low).convert("RGB"))

        ela = np.abs(image_uint8.astype(np.float32) - img_low.astype(np.float32))
        ela = ela.mean(axis=2)
        ela = ela / (ela.max() + 1e-8)
        ela_map = ela.astype(np.float32)
        # ela = np.clip(ela * 3.0, 0, 1)

        logger.debug("ELA min/max: %.6f/%.6f", float(ela_map.min()), float(ela_map.max()))
        return ela_map
    except Exception as exc:
        logger.warning("Failed to compute ELA for image: %s", exc)
        try:
            height, width = image_rgb.shape[:2]
        except Exception:
            height, width = 0, 0
        return np.zeros((height, width), dtype=np.float32)


def compute_c8_authenticity_score(image_rgb: np.ndarray,
                                  image_gray: np.ndarray) -> dict:
    """
    Compute global authenticity features for C8 detection.
    """
    try:
        if image_rgb is None or image_gray is None:
            raise ValueError("image_rgb or image_gray is None")

        gray_uint8 = image_gray.astype(np.uint8, copy=False)

        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        sigma_lap = float(laplacian.std())

        _, thresh = cv2.threshold(
            gray_uint8,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        edges = cv2.Canny(thresh, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )

        angles: list[float] = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                angle = float(np.degrees(np.arctan2(dy, dx)))
                if -45.0 <= angle <= 45.0:
                    angles.append(angle)

        if angles:
            theta_skew = float(np.median(angles))
        else:
            theta_skew = 0.0

        height, width = gray_uint8.shape[:2]
        kappa_dct = 0.0
        if height >= 8 and width >= 8:
            gray_float = gray_uint8.astype(np.float32)
            coeffs: list[np.ndarray] = []
            for y in range(0, height - 7, 8):
                for x in range(0, width - 7, 8):
                    block = gray_float[y:y + 8, x:x + 8]
                    dct_block = dctn(block)
                    flat = dct_block.flatten()
                    if flat.size > 1:
                        coeffs.append(flat[1:])

            if coeffs:
                all_coeffs = np.concatenate(coeffs)
                try:
                    kappa_val = float(kurtosis(all_coeffs, fisher=True, bias=False))
                    if np.isfinite(kappa_val):
                        kappa_dct = kappa_val
                except Exception:
                    kappa_dct = 0.0

        _, thresh_bg = cv2.threshold(
            gray_uint8,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        bg_mask = thresh_bg == 0
        background_pixels = gray_uint8[bg_mask]
        if background_pixels.size == 0:
            entropy_bg = 0.0
        else:
            entropy_bg = float(shannon_entropy(background_pixels.reshape(-1, 1)))

        is_ai_generated = (
            sigma_lap < C8_SIGMA_LAP_THRESHOLD
            and entropy_bg < C8_ENTROPY_BG_THRESHOLD
        )

        logger.debug(
            f"C8 features | sigma_lap={sigma_lap:.3f}, "
            f"theta_skew={theta_skew:.3f}, "
            f"kappa_dct={kappa_dct:.3f}, "
            f"entropy_bg={entropy_bg:.3f}"
        )
        if is_ai_generated:
            logger.info(
                "C8 detected: sigma_lap=%.6f entropy_bg=%.6f",
                sigma_lap,
                entropy_bg,
            )

        return {
            "sigma_lap": sigma_lap,
            "theta_skew": theta_skew,
            "kappa_dct": kappa_dct,
            "entropy_bg": entropy_bg,
            "is_ai_generated": is_ai_generated,
        }
    except Exception as exc:
        logger.warning("Failed to compute C8 authenticity score: %s", exc)
        return {
            "sigma_lap": 0.0,
            "theta_skew": 0.0,
            "kappa_dct": 0.0,
            "entropy_bg": 0.0,
            "is_ai_generated": False,
        }


def detect_irregular_spacing(image_rgb: np.ndarray) -> list[dict]:
    """
    Detect lines with irregular inter-word spacing using Tesseract OCR.
    """
    try:
        if image_rgb is None:
            return []

        try:
            data = pytesseract.image_to_data(
                image_rgb,
                output_type=pytesseract.Output.DICT,
                config="--psm 6 --oem 1",
            )
        except Exception as exc:
            logger.warning("Failed to run OCR for spacing detection: %s", exc)
            return []

        if not data or "text" not in data:
            return []

        words: list[dict] = []
        count = len(data.get("text", []))
        for i in range(count):
            text = data["text"][i]
            if text is None or str(text).strip() == "":
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                continue
            if conf == -1 or conf < 30:
                continue
            try:
                left = int(data["left"][i])
                top = int(data["top"][i])
                width = int(data["width"][i])
                height = int(data["height"][i])
            except Exception:
                continue

            words.append(
                {
                    "text": str(text),
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "conf": conf,
                }
            )

        if not words:
            return []

        avg_word_height = float(np.mean([w["height"] for w in words])) if words else 0.0
        line_threshold = max(5, int(0.5 * avg_word_height))

        words_sorted = sorted(words, key=lambda w: (w["top"], w["left"]))
        lines: list[dict] = []
        for word in words_sorted:
            matched_line = None
            min_delta = None
            for line in lines:
                delta = abs(word["top"] - line["top"])
                if delta <= line_threshold and (min_delta is None or delta < min_delta):
                    matched_line = line
                    min_delta = delta
            if matched_line is None:
                lines.append({"top": word["top"], "words": [word]})
                # line["top"] = int(np.mean([w["top"] for w in line["words"]]))
            else:
                matched_line["words"].append(word)

        results: list[dict] = []
        total_gaps = 0
        for line in lines:
            line_words = sorted(line["words"], key=lambda w: w["left"])
            # lines.sort(key=lambda l: l["top"])
            if len(line_words) < 3:
                continue

            gap_entries: list[tuple[float, dict, dict]] = []
            for idx in range(len(line_words) - 1):
                curr = line_words[idx]
                nxt = line_words[idx + 1]
                gap = nxt["left"] - (curr["left"] + curr["width"])
                if gap <= 0:
                    continue
                gap_entries.append((float(gap), curr, nxt))

            if not gap_entries:
                continue

            gaps = np.array([entry[0] for entry in gap_entries], dtype=np.float32)
            total_gaps += len(gaps)

            median_gap = float(np.median(gaps))
            q75 = float(np.percentile(gaps, 75))
            q25 = float(np.percentile(gaps, 25))
            iqr = q75 - q25
            if iqr <= 0:
                continue

            threshold = median_gap + C7_IQR_MULTIPLIER * iqr
            for gap, curr, nxt in gap_entries:
                if gap > threshold:
                    stretch_factor = gap / (median_gap + 1e-8)
                    if stretch_factor < C7_MIN_STRETCH_FACTOR:
                        continue

                    x = curr["left"]
                    y = min(curr["top"], nxt["top"])
                    w = (nxt["left"] + nxt["width"]) - x
                    h = max(curr["top"] + curr["height"], nxt["top"] + nxt["height"]) - y
                    if w <= 0 or h <= 0:
                        continue
                    results.append(
                        {
                            "x": int(x),
                            "y": int(y),
                            "w": int(w),
                            "h": int(h),
                            "stretch_factor": float(stretch_factor),
                        }
                    )

        if results:
            filtered: list[dict] = []
            for candidate in results:
                overlap_found = False
                for kept in filtered:
                    x_left = max(candidate["x"], kept["x"])
                    y_top = max(candidate["y"], kept["y"])
                    x_right = min(candidate["x"] + candidate["w"], kept["x"] + kept["w"])
                    y_bottom = min(candidate["y"] + candidate["h"], kept["y"] + kept["h"])
                    if x_right <= x_left or y_bottom <= y_top:
                        continue
                    inter_area = (x_right - x_left) * (y_bottom - y_top)
                    area_candidate = candidate["w"] * candidate["h"]
                    area_kept = kept["w"] * kept["h"]
                    if area_candidate <= 0 or area_kept <= 0:
                        continue
                    overlap_ratio = inter_area / float(min(area_candidate, area_kept))
                    if overlap_ratio > 0.8:
                        overlap_found = True
                        break
                if not overlap_found:
                    filtered.append(candidate)
            results = filtered

        logger.debug(
            f"C7 spacing: lines={len(lines)}, gaps={total_gaps}, flagged={len(results)}"
        )
        return results
    except Exception as exc:
        logger.warning("Failed to detect irregular spacing: %s", exc)
        return []


def detect_overwriting(image_rgb: np.ndarray,
                       image_gray: np.ndarray) -> np.ndarray:
    """
    Detect overwritten regions using intensity and edge inconsistencies.
    """
    try:
        if image_gray is None:
            logger.warning("Failed to detect overwriting: image_gray is None")
            if image_rgb is not None:
                height, width = image_rgb.shape[:2]
            else:
                height, width = 0, 0
            return np.zeros((height, width), dtype=np.float32)

        height, width = image_gray.shape[:2]
        gray_uint8 = image_gray.astype(np.uint8, copy=False)

        blur = cv2.GaussianBlur(gray_uint8, (15, 15), 0)
        diff = cv2.absdiff(gray_uint8, blur)
        diff = diff.astype(np.float32) / 255.0

        edges = cv2.Canny(gray_uint8, 50, 150)
        edges = edges.astype(np.float32) / 255.0

        heatmap = 0.6 * diff + 0.4 * edges
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap = heatmap.astype(np.float32)

        logger.debug("C2 heatmap min/max: %.6f/%.6f", float(heatmap.min()), float(heatmap.max()))
        return heatmap
    except Exception as exc:
        logger.warning("Failed to detect overwriting: %s", exc)
        try:
            height, width = image_gray.shape[:2]
        except Exception:
            height, width = 0, 0
        return np.zeros((height, width), dtype=np.float32)


def detect_field_noise_outliers(image_rgb: np.ndarray,
                                image_gray: np.ndarray) -> np.ndarray:
    """
    Detect field-level noise outliers using local variance and edge density.
    """
    try:
        if image_gray is None:
            logger.warning("Failed to detect field noise outliers: image_gray is None")
            if image_rgb is not None:
                height, width = image_rgb.shape[:2]
            else:
                height, width = 0, 0
            return np.zeros((height, width), dtype=np.float32)

        height, width = image_gray.shape[:2]
        gray_uint8 = image_gray.astype(np.uint8, copy=False)
        gray_float = gray_uint8.astype(np.float32)

        mean = cv2.blur(gray_float, (15, 15))
        sq_mean = cv2.blur(gray_float ** 2, (15, 15))
        variance = sq_mean - mean ** 2
        variance = np.maximum(variance, 0)
        variance = variance / (variance.max() + 1e-8)

        edges = cv2.Canny(gray_uint8, 50, 150)
        edges = edges.astype(np.float32) / 255.0
        edge_density = cv2.blur(edges, (15, 15))

        heatmap = 0.7 * variance + 0.3 * edge_density
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap = heatmap.astype(np.float32)

        logger.debug("C9 heatmap min/max: %.6f/%.6f", float(heatmap.min()), float(heatmap.max()))
        return heatmap
    except Exception as exc:
        logger.warning("Failed to detect field noise outliers: %s", exc)
        try:
            height, width = image_gray.shape[:2]
        except Exception:
            height, width = 0, 0
        return np.zeros((height, width), dtype=np.float32)


def compute_inter_band_features(image_gray: np.ndarray) -> dict:
    """
    Compute horizontal band transition features for C5 detection.
    """
    try:
        if image_gray is None:
            return {
                "band_transitions": [],
                "num_transitions": 0,
                "is_suspicious": False,
            }

        gray = image_gray.astype(np.float32)
        height, width = gray.shape[:2]

        band_height = 32
        num_bands = height // band_height
        if num_bands < 2:
            logger.debug("C5 bands: count=0 transitions=0")
            return {
                "band_transitions": [],
                "num_transitions": 0,
                "is_suspicious": False,
            }

        band_means: list[float] = []
        band_vars: list[float] = []
        for band_idx in range(num_bands):
            y_start = band_idx * band_height
            y_end = y_start + band_height
            band = gray[y_start:y_end, :]
            band_means.append(float(np.mean(band)))
            band_vars.append(float(np.var(band)))

        band_transitions: list[int] = []
        for idx in range(1, num_bands):
            delta_mean = abs(band_means[idx] - band_means[idx - 1])
            delta_var = abs(band_vars[idx] - band_vars[idx - 1])
            if delta_mean > 15 or delta_var > 20:
                band_transitions.append(idx * band_height)

        num_transitions = len(band_transitions)
        is_suspicious = num_transitions >= 2

        logger.debug(
            "C5 bands: count=%d transitions=%d",
            num_bands,
            num_transitions,
        )

        return {
            "band_transitions": band_transitions,
            "num_transitions": num_transitions,
            "is_suspicious": is_suspicious,
        }
    except Exception as exc:
        logger.warning("Failed to compute inter-band features: %s", exc)
        return {
            "band_transitions": [],
            "num_transitions": 0,
            "is_suspicious": False,
        }
