from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Tuple, Dict, List

from PIL import Image

# -------------------------
# Batch folder processing
# GATE A: confidence & geometric sanity
# - Top-K per prompt per immagine
# - Soglia grounding: score_box >= τ_box
# - Area ratio (box): area_box/area_img in [0.01, 0.5]
# - Aspect ratio: w/h in [0.2, 5]
# -------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

@dataclass
class FilterConfig:
    mode: str                 # "GATE A" | others to be implemented
    top_k: int = 5
    grounding_threshold: float = 0.5
    area_ratio_range: Tuple[float, float] = (0.01, 0.5)
    aspect_ratio_range: Tuple[float, float] = (0.2, 5.0)

def filter_detections(
    detections: Dict[str, List[Dict]],
    cfg: FilterConfig,
) -> Dict[str, List[Dict]]:
    m = cfg.mode.upper()
    if m == "GATE A":
        return _gate_a_filter(detections, cfg)
    raise ValueError(f"Unknown mode: {cfg.mode}")

def _gate_a_filter(
    detections: Dict[str, List[Dict]],
    cfg: FilterConfig,
) -> Dict[str, List[Dict]]:
    filtered = {}
    for prompt, dets in detections.items():
        # Sort by confidence score and take top-K
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)[:cfg.top_k]
        valid_dets = []
        for d in dets:
            score_box = d["score"]
            area_box = d["bbox"][2] * d["bbox"][3]  # w * h
            area_img = d["img_width"] * d["img_height"]
            aspect_ratio = d["bbox"][2] / max(d["bbox"][3], 1e-6)  # w/h

            if (score_box >= cfg.grounding_threshold and
                cfg.area_ratio_range[0] <= area_box / area_img <= cfg.area_ratio_range[1] and
                cfg.aspect_ratio_range[0] <= aspect_ratio <= cfg.aspect_ratio_range[1]):
                valid_dets.append(d)
        filtered[prompt] = valid_dets
    return filtered

def _iter_images(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(dirpath, fn))
    return paths

def process_folder(
    in_root: str,
    out_root: str,
    cfg: FilterConfig,
):
    os.makedirs(out_root, exist_ok=True)
    img_paths = _iter_images(in_root)
    for img_path in img_paths:
        with Image.open(img_path) as img:
            detections = {}  # This should be replaced with actual loading logic
            filtered_dets = filter_detections(detections, cfg)
            # For example, save to a JSON file in out_root
            out_path = os.path.join(out_root, os.path.basename(img_path) + ".json")
            with open(out_path, "w") as f:
                json.dump(filtered_dets, f)

def main():
    cfg = FilterConfig(mode="GATE A", top_k=5, grounding_threshold=0.5)
    process_folder(r"C:\Users\cogol\PycharmProjects\ACG4CAV\data\images\zebra", r"C:\Users\cogol\PycharmProjects\ACG4CAV\data\images\raw", cfg)

if __name__ == "__main__":
    main()
