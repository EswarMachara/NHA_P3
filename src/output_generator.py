import json
import os

import yaml

from src.config import get_logger


logger = get_logger(__name__)


def write_yaml(page_info: dict,
               category_results: list[dict],
               output_dir: str) -> str:
    """
    Write a YAML annotation file for a processed document page.
    """
    try:
        if not page_info or not category_results:
            return None

        categories = [
            result.get("category")
            for result in category_results
            if result.get("category")
        ]
        if categories and all(cat in ["C8", "C10"] for cat in categories):
            return None

        file_name = page_info.get("file_name")
        if not file_name:
            logger.warning("Missing file_name for YAML output")
            return None

        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(file_name)[0]
        yaml_path = os.path.join(output_dir, f"{stem}.yaml")

        entries: list[dict] = []
        for result in category_results:
            category = result.get("category")
            if category in {"C8", "C10", None}:
                continue

            bboxes = result.get("bboxes") or []
            extra = result.get("extra") or {}

            if category in {"C1", "C2", "C6"}:
                for bbox in bboxes:
                    entries.append(
                        {
                            "h": int(bbox["h"]),
                            "w": int(bbox["w"]),
                            "x": int(bbox["x"]),
                            "y": int(bbox["y"]),
                        }
                    )
            elif category == "C3":
                types = extra.get("types", [])
                for idx, bbox in enumerate(bboxes):
                    type_value = None
                    if isinstance(types, list) and idx < len(types):
                        type_value = types[idx]
                    elif "type" in extra:
                        type_value = extra.get("type")
                    if not type_value:
                        logger.warning("Missing C3 type for %s index %d", file_name, idx)
                        continue
                    entries.append(
                        {
                            "h": int(bbox["h"]),
                            "type": str(type_value),
                            "w": int(bbox["w"]),
                            "x": int(bbox["x"]),
                            "y": int(bbox["y"]),
                        }
                    )
            elif category == "C4":
                for bbox in bboxes:
                    entries.append(
                        {
                            "h": int(bbox["h"]),
                            "type": "erased",
                            "w": int(bbox["w"]),
                            "x": int(bbox["x"]),
                            "y": int(bbox["y"]),
                        }
                    )
            elif category == "C5":
                types = extra.get("types", [])
                for idx, bbox in enumerate(bboxes):
                    type_value = None
                    if isinstance(types, list) and idx < len(types):
                        type_value = types[idx]
                    elif "type" in extra:
                        type_value = extra.get("type")
                    if type_value not in {"body", "header"}:
                        logger.warning("Missing C5 type for %s index %d", file_name, idx)
                        type_value = "body"

                    if type_value == "body":
                        body_source = "other"
                        header_source = file_name
                    else:
                        body_source = file_name
                        header_source = "other"

                    entries.append(
                        {
                            "body_source": str(body_source),
                            "h": int(bbox["h"]),
                            "header_source": str(header_source),
                            "type": str(type_value),
                            "w": int(bbox["w"]),
                            "x": int(bbox["x"]),
                            "y": int(bbox["y"]),
                        }
                    )
            elif category == "C7":
                stretch_values = extra.get("stretch_factors", [])
                for idx, bbox in enumerate(bboxes):
                    stretch_factor = None
                    if isinstance(bbox, dict) and "stretch_factor" in bbox:
                        stretch_factor = bbox.get("stretch_factor")
                    elif isinstance(stretch_values, list) and idx < len(stretch_values):
                        stretch_factor = stretch_values[idx]
                    elif "stretch_factor" in extra:
                        stretch_factor = extra.get("stretch_factor")
                    if stretch_factor is None:
                        logger.warning("Missing C7 stretch_factor for %s index %d", file_name, idx)
                        stretch_factor = 0.0
                    entries.append(
                        {
                            "h": int(bbox["h"]),
                            "stretch_factor": round(float(stretch_factor), 3),
                            "type": "irregular_spacing",
                            "w": int(bbox["w"]),
                            "x": int(bbox["x"]),
                            "y": int(bbox["y"]),
                        }
                    )
            elif category == "C9":
                types = extra.get("types", [])
                new_vals = extra.get("new", [])
                old_vals = extra.get("old", [])
                for idx, bbox in enumerate(bboxes):
                    type_value = None
                    if isinstance(types, list) and idx < len(types):
                        type_value = types[idx]
                    elif "type" in extra:
                        type_value = extra.get("type")
                    if type_value is None:
                        logger.warning("Missing C9 type for %s index %d", file_name, idx)
                        type_value = "name"

                    if isinstance(new_vals, list) and idx < len(new_vals):
                        new_val = new_vals[idx]
                    else:
                        new_val = extra.get("new", "")

                    if isinstance(old_vals, list) and idx < len(old_vals):
                        old_val = old_vals[idx]
                    else:
                        old_val = extra.get("old", "")

                    entries.append(
                        {
                            "h": int(bbox["h"]),
                            "new": str(new_val),
                            "old": str(old_val),
                            "type": str(type_value),
                            "w": int(bbox["w"]),
                            "x": int(bbox["x"]),
                            "y": int(bbox["y"]),
                        }
                    )

        with open(yaml_path, "w", encoding="utf-8") as handle:
            yaml.dump(
                entries,
                handle,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        logger.info("Wrote YAML: %s", yaml_path)
        return yaml_path
    except Exception as exc:
        logger.warning("Failed to write YAML: %s", exc)
        return None


def write_json(all_page_results: list[dict],
               output_dir: str) -> str:
    """
    Write submissions.json for all processed pages.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "submissions.json")

        entries: list[dict] = []
        for result in all_page_results:
            categories = result.get("categories") or []
            if isinstance(categories, str):
                category_id = categories
            else:
                ordered = [cat for cat in categories if cat]
                category_id = "||".join(ordered)
            if not category_id:
                category_id = "C10"

            entries.append(
                {
                    "link": result.get("link", ""),
                    "file_name": result.get("file_name", ""),
                    "Category_ID": category_id,
                }
            )

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2, ensure_ascii=False)

        logger.info("Wrote JSON: %s", json_path)
        return json_path
    except Exception as exc:
        logger.warning("Failed to write JSON: %s", exc)
        return None
