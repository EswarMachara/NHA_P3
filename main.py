import argparse
import logging
import os

from category_assignment import assign_categories
from config import THRESHOLD, get_logger
from output_generator import write_json, write_yaml
from preprocessor import preprocess_file
from streams.catnet_stream import load_catnet_model, run_catnet
from streams.statistical_stream import (
    compute_c8_authenticity_score,
    compute_ela,
    compute_inter_band_features,
    detect_field_noise_outliers,
    detect_irregular_spacing,
    detect_overwriting,
)
from streams.trufor_stream import load_trufor_model, run_trufor


def main() -> None:
    parser = argparse.ArgumentParser(description="DAFD pipeline")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = get_logger(__name__)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    try:
        trufor_model = load_trufor_model(device=args.device)
        catnet_model = load_catnet_model(device=args.device)
    except Exception as exc:
        logger.warning("Failed to load models: %s", exc)
        return

    all_page_results: list[dict] = []

    for entry in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, entry)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".pdf"}:
            continue

        logger.info("Processing file: %s", file_path)
        try:
            pages = preprocess_file(file_path)
        except Exception as exc:
            logger.warning("Preprocessing failed for %s: %s", file_path, exc)
            continue

        if not pages:
            logger.warning("No pages returned for %s", file_path)
            continue

        for page in pages:
            try:
                page_name = page.get("file_name", "")
                logger.debug("Processing page: %s", page_name)

                ela_map = compute_ela(page["image_rgb"])
                c8_info = compute_c8_authenticity_score(page["image_rgb"], page["image_gray"])
                c7_regions = detect_irregular_spacing(page["image_rgb"])
                c2_map = detect_overwriting(page["image_rgb"], page["image_gray"])
                c9_map = detect_field_noise_outliers(page["image_rgb"], page["image_gray"])
                c5_info = compute_inter_band_features(page["image_gray"])

                trufor_map, _ = run_trufor(trufor_model, page["image_rgb"], device=args.device)
                catnet_map = run_catnet(catnet_model, page["image_rgb"], page["is_jpeg"], device=args.device)

                heatmaps = {
                    "ela": ela_map,
                    "c2": c2_map,
                    "c9": c9_map,
                    "trufor": trufor_map,
                    "catnet": catnet_map,
                }
                statistical_results = {
                    "c7": c7_regions,
                    "c8": c8_info,
                    "c5": c5_info,
                }

                category_results = assign_categories(page, heatmaps, statistical_results, THRESHOLD)

                write_yaml(
                    {
                        "file_name": page.get("file_name"),
                        "file_path": page.get("file_path"),
                    },
                    category_results,
                    output_dir,
                )

                categories = [result.get("category") for result in category_results if result.get("category")]
                all_page_results.append(
                    {
                        "link": page.get("file_path", ""),
                        "file_name": page.get("file_name", ""),
                        "Category_ID": "||".join(categories) if categories else "C10",
                    }
                )
            except Exception as exc:
                logger.warning("Failed to process page %s: %s", page.get("file_name"), exc)
                continue

    write_json(all_page_results, output_dir)


if __name__ == "__main__":
    main()
