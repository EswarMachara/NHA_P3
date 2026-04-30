import base64
import io
import os
from functools import lru_cache

import numpy as np
import yaml
from IPython.display import HTML, display
from PIL import Image, ImageDraw
import ipywidgets as widgets

from src.preprocessor import preprocess_file


def _list_input_files(input_dir: str) -> list[str]:
    if not os.path.isdir(input_dir):
        return []
    files = []
    for name in os.listdir(input_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".pdf"}:
            files.append(name)
    return sorted(files)


@lru_cache(maxsize=32)
def _load_pages(file_path: str) -> list[dict]:
    return preprocess_file(file_path)


def _load_yaml_entries(output_dir: str, page_file_name: str) -> list[dict]:
    stem = os.path.splitext(page_file_name)[0]
    yaml_path = os.path.join(output_dir, f"{stem}.yaml")
    if not os.path.exists(yaml_path):
        return []
    with open(yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if isinstance(data, list):
        return data
    return []


def _encode_png(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _render_side_by_side(original: Image.Image, overlay: Image.Image, display_width: int) -> None:
        original_b64 = _encode_png(original)
        overlay_b64 = _encode_png(overlay)
        html = f"""
        <div style="display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="font-weight:600; margin-bottom:6px;">Original</div>
                <img src="data:image/png;base64,{original_b64}" style="width:{display_width}px; height:auto;" />
            </div>
            <div style="text-align:center;">
                <div style="font-weight:600; margin-bottom:6px;">Overlay</div>
                <img src="data:image/png;base64,{overlay_b64}" style="width:{display_width}px; height:auto;" />
            </div>
        </div>
        """
        display(HTML(html))


def _infer_category(entry: dict) -> str:
    entry_type = entry.get("type")
    if entry_type == "irregular_spacing" or "stretch_factor" in entry:
        return "C7"
    if entry_type == "erased":
        return "C4"
    if entry_type in {"text", "stamp", "signature"}:
        return "C3"
    if entry_type in {"body", "header"} or "body_source" in entry or "header_source" in entry:
        return "C5"
    if entry_type in {"name", "date", "amount"} or "new" in entry or "old" in entry:
        return "C9"
    return "C1/C2/C6"


def _label_for_entry(entry: dict, category: str) -> str:
    entry_type = entry.get("type")
    if category == "C7":
        stretch = entry.get("stretch_factor")
        if stretch is not None:
            return f"C7 {stretch:.2f}"
        return "C7"
    if category == "C9" and entry_type:
        return f"C9 {entry_type}"
    if category == "C3" and entry_type:
        return f"C3 {entry_type}"
    if category == "C5" and entry_type:
        return f"C5 {entry_type}"
    if category == "C4":
        return "C4"
    return category


def _draw_overlay(
    image_rgb: np.ndarray,
    entries: list[dict],
    fill_alpha: int,
    outline_alpha: int,
    line_width: int,
    show_labels: bool,
) -> Image.Image:
    base = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    colors = {
        "C1/C2/C6": (255, 140, 0),
        "C3": (220, 20, 60),
        "C4": (128, 0, 128),
        "C5": (0, 128, 255),
        "C7": (0, 200, 100),
        "C9": (255, 215, 0),
    }

    for entry in entries:
        try:
            x = int(entry["x"])
            y = int(entry["y"])
            w = int(entry["w"])
            h = int(entry["h"])
        except Exception:
            continue

        category = _infer_category(entry)
        label = _label_for_entry(entry, category)
        color = colors.get(category, (255, 0, 0))
        fill = (color[0], color[1], color[2], fill_alpha)
        outline = (color[0], color[1], color[2], outline_alpha)

        draw.rectangle([x, y, x + w, y + h], outline=outline, width=line_width, fill=fill)
        if show_labels:
            draw.text((x + 2, y + 2), label, fill=(255, 255, 255, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


def launch_viewer(input_dir: str, output_dir: str) -> None:
    files = _list_input_files(input_dir)
    if not files:
        print(f"No supported files found in {input_dir}")
        return

    file_dropdown = widgets.Dropdown(options=files, description="File")
    page_slider = widgets.IntSlider(value=1, min=1, max=1, step=1, description="Page")
    category_select = widgets.SelectMultiple(
        options=("C1/C2/C6", "C3", "C4", "C5", "C7", "C9"),
        value=("C1/C2/C6", "C3", "C4", "C5", "C7", "C9"),
        description="Cats",
    )
    fill_alpha_slider = widgets.IntSlider(value=40, min=0, max=120, step=5, description="Fill")
    outline_alpha_slider = widgets.IntSlider(value=220, min=60, max=255, step=5, description="Outline")
    line_width_slider = widgets.IntSlider(value=2, min=1, max=6, step=1, description="Line")
    outline_only = widgets.Checkbox(value=False, description="Outline only")
    show_labels = widgets.Checkbox(value=True, description="Labels")
    max_area_slider = widgets.IntSlider(value=100, min=5, max=100, step=5, description="Max%")
    display_width_slider = widgets.IntSlider(value=900, min=300, max=1800, step=50, description="Width")
    out = widgets.Output()

    def update_view(*_args) -> None:
        with out:
            out.clear_output(wait=True)
            file_name = file_dropdown.value
            file_path = os.path.join(input_dir, file_name)
            pages = _load_pages(file_path)
            if not pages:
                print("No pages available.")
                return

            page_index = min(page_slider.value - 1, len(pages) - 1)
            page = pages[page_index]
            image_rgb = page["image_rgb"]
            entries = _load_yaml_entries(output_dir, page["file_name"])
            selected = set(category_select.value)
            filtered = []
            height, width = image_rgb.shape[:2]
            max_ratio = max_area_slider.value / 100.0
            max_area = max_ratio * height * width
            for entry in entries:
                category = _infer_category(entry)
                if category not in selected:
                    continue
                try:
                    area = int(entry["w"]) * int(entry["h"])
                except Exception:
                    area = None
                if area is not None and area > max_area:
                    continue
                filtered.append(entry)

            fill_alpha = 0 if outline_only.value else fill_alpha_slider.value
            overlay = _draw_overlay(
                image_rgb,
                filtered,
                fill_alpha=fill_alpha,
                outline_alpha=outline_alpha_slider.value,
                line_width=line_width_slider.value,
                show_labels=show_labels.value,
            )

            original_pil = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
            _render_side_by_side(original_pil, overlay, display_width_slider.value)

    def on_file_change(change) -> None:
        if change.get("name") != "value":
            return
        file_name = file_dropdown.value
        file_path = os.path.join(input_dir, file_name)
        pages = _load_pages(file_path)
        page_slider.max = max(1, len(pages))
        page_slider.value = 1
        page_slider.disabled = len(pages) <= 1
        update_view()

    file_dropdown.observe(on_file_change, names="value")
    page_slider.observe(update_view, names="value")
    category_select.observe(update_view, names="value")
    fill_alpha_slider.observe(update_view, names="value")
    outline_alpha_slider.observe(update_view, names="value")
    line_width_slider.observe(update_view, names="value")
    outline_only.observe(update_view, names="value")
    show_labels.observe(update_view, names="value")
    max_area_slider.observe(update_view, names="value")
    display_width_slider.observe(update_view, names="value")

    on_file_change({"name": "value"})
    controls_row1 = widgets.HBox([file_dropdown, page_slider, display_width_slider])
    controls_row2 = widgets.HBox([
        category_select,
        widgets.VBox([fill_alpha_slider, outline_alpha_slider, line_width_slider]),
        widgets.VBox([outline_only, show_labels, max_area_slider]),
    ])
    display(controls_row1)
    display(controls_row2)
    display(out)
