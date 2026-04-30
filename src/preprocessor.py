import os

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

from src.config import PDF_DPI, IMAGE_MAX_DIM, get_logger


logger = get_logger(__name__)


def pdf_to_images(pdf_path: str, dpi: int = PDF_DPI) -> list[np.ndarray]:
    """
    Convert all pages of a PDF to RGB numpy arrays at the specified DPI.
    """
    try:
        pil_pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as exc:
        logger.warning("Failed to convert PDF %s: %s", pdf_path, exc)
        return []

    images: list[np.ndarray] = []
    for page_num, page in enumerate(pil_pages, start=1):
        if page is None:
            logger.warning("Empty PDF page %d in %s", page_num, pdf_path)
            continue
        try:
            page_rgb = page.convert("RGB")
            page_array = np.array(page_rgb)
            if page_array.size == 0:
                logger.warning("Empty PDF page %d in %s", page_num, pdf_path)
                continue
            if page_array.dtype != np.uint8:
                page_array = page_array.astype(np.uint8)
            images.append(page_array)
        except Exception as exc:
            logger.warning("Failed to process PDF page %d in %s: %s", page_num, pdf_path, exc)
            continue

    return images


def normalize_image(image: np.ndarray, max_dim: int = IMAGE_MAX_DIM) -> np.ndarray:
    """
    Resize image if its largest dimension exceeds max_dim, preserving aspect ratio.
    """
    if image is None:
        raise ValueError("normalize_image received None")

    height, width = image.shape[:2]
    if max(height, width) <= max_dim:
        return image.astype(np.uint8, copy=False)

    scale = max_dim / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    logger.debug(
        "Resizing image from (%d, %d) to (%d, %d)",
        height,
        width,
        new_height,
        new_width,
    )
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8, copy=False)


def preprocess_file(file_path: str) -> list[dict]:
    """
    Accept a single input file and return a list of page dictionaries.
    """
    logger.info("Preprocessing file: %s", file_path)

    if not file_path:
        logger.warning("Empty file path provided")
        return []

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".pdf"}:
        logger.warning("Unsupported file type: %s", file_path)
        return []

    results: list[dict] = []

    if ext == ".pdf":
        pages = pdf_to_images(file_path, dpi=PDF_DPI)
        if not pages:
            logger.warning("No pages extracted from PDF: %s", file_path)
            logger.info("Finished preprocessing file: %s (0 pages)", file_path)
            return []

        stem = os.path.splitext(os.path.basename(file_path))[0]
        for page_num, rgb in enumerate(pages, start=1):
            if rgb is None or rgb.size == 0:
                logger.warning("Empty PDF page %d in %s", page_num, file_path)
                continue

            try:
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    logger.warning("Invalid PDF page shape %s in %s", rgb.shape, file_path)
                    continue

                original_h, original_w = rgb.shape[:2]
                logger.debug("PDF page %d shape: %s", page_num, rgb.shape)

                normalized = normalize_image(rgb, max_dim=IMAGE_MAX_DIM)
                gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)

                if gray is None or gray.size == 0:
                    logger.warning("Failed grayscale conversion for PDF page %d in %s", page_num, file_path)
                    continue

                file_name = f"{stem}_page_{page_num}.jpg"
                results.append(
                    {
                        "image_rgb": normalized,
                        "image_gray": gray,
                        "original_h": int(original_h),
                        "original_w": int(original_w),
                        "file_path": file_path,
                        "file_name": file_name,
                        "format": "pdf",
                        "page_num": int(page_num),
                        "is_jpeg": False,
                    }
                )
            except Exception as exc:
                logger.warning("Failed to process PDF page %d in %s: %s", page_num, file_path, exc)
                continue
    else:
        is_jpeg = ext in {".jpg", ".jpeg"}
        file_name = os.path.basename(file_path)
        file_format = "jpg" if is_jpeg else "png"

        try:
            with Image.open(file_path) as img:
                rgb = np.array(img.convert("RGB"))
        except Exception as exc:
            logger.warning("Failed to load image %s: %s", file_path, exc)
            logger.info("Finished preprocessing file: %s (0 pages)", file_path)
            return []

        if rgb is None or rgb.size == 0:
            logger.warning("Empty image data for %s", file_path)
            logger.info("Finished preprocessing file: %s (0 pages)", file_path)
            return []

        try:
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                logger.warning("Invalid image shape %s in %s", rgb.shape, file_path)
                logger.info("Finished preprocessing file: %s (0 pages)", file_path)
                return []

            original_h, original_w = rgb.shape[:2]
            logger.debug("Image shape: %s", rgb.shape)

            normalized = normalize_image(rgb, max_dim=IMAGE_MAX_DIM)
            gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)

            if gray is None or gray.size == 0:
                logger.warning("Failed grayscale conversion for %s", file_path)
                logger.info("Finished preprocessing file: %s (0 pages)", file_path)
                return []

            results.append(
                {
                    "image_rgb": normalized,
                    "image_gray": gray,
                    "original_h": int(original_h),
                    "original_w": int(original_w),
                    "file_path": file_path,
                    "file_name": file_name,
                    "format": file_format,
                    "page_num": 1,
                    "is_jpeg": is_jpeg,
                }
            )
        except Exception as exc:
            logger.warning("Failed to process image %s: %s", file_path, exc)

    logger.info("Finished preprocessing file: %s (%d pages)", file_path, len(results))
    return results
