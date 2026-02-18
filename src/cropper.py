# Cropper adapted to consume GrounderSAM2 outputs (segmentations and/or boxes)
# - supports per-image MULTIPLE detections
# - can crop from: masks, boxes, or "mask->bbox fallback to box"
# - preserves folder structure, writes *_det{i}.ext when multiple crops are produced

from __future__ import annotations

import os
import json
import numpy as np

from dataclasses import dataclass
from PIL import Image
from typing import Optional, Tuple, Dict, List, Any, Union

from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# -------------------------
# Helpers
# -------------------------

def _with_det_suffix(out_path: str, i: int) -> str:
    p = Path(out_path)
    return str(p.with_name(f"{p.stem}_det{i}{p.suffix}"))

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _square_crop_box(cx: float, cy: float, side: float, w: int, h: int) -> Tuple[int, int, int, int]:
    side = max(1.0, side)
    half = side / 2.0

    left = cx - half
    top = cy - half
    right = cx + half
    bottom = cy + half

    # shift inside bounds
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > w:
        left -= (right - w)
        right = w
    if bottom > h:
        top -= (bottom - h)
        bottom = h

    # final clamp
    left = _clamp(int(round(left)), 0, w)
    top = _clamp(int(round(top)), 0, h)
    right = _clamp(int(round(right)), 0, w)
    bottom = _clamp(int(round(bottom)), 0, h)

    # ensure valid
    if right <= left:
        right = min(w, left + 1)
    if bottom <= top:
        bottom = min(h, top + 1)

    return left, top, right, bottom

def center_crop(img: Image.Image, out_size: int) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    cx, cy = w / 2.0, h / 2.0
    box = _square_crop_box(cx, cy, side, w, h)
    cropped = img.crop(box)
    return cropped.resize((out_size, out_size), resample=Image.BICUBIC)

def bbox_crop(
    img: Image.Image,
    out_size: int,
    bbox_xyxy: Tuple[float, float, float, float],
    padding: float = 0.1,
) -> Image.Image:
    """
    bbox_xyxy in pixel coords: (x1,y1,x2,y2).
    padding expands bbox by fraction of max(bw,bh), then makes it square.
    """
    w, h = img.size
    x1, y1, x2, y2 = bbox_xyxy
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + float(padding))

    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    box = _square_crop_box(cx, cy, side, w, h)
    cropped = img.crop(box)
    return cropped.resize((out_size, out_size), resample=Image.BICUBIC)

def find_biggest_bbox_inside_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Largest axis-aligned rectangle fully inside a binary mask (O(H*W)).
    Returns (x1,y1,x2,y2) in pixel coords, where (x2,y2) is exclusive.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W), got {mask.shape}")

    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    H, W = m.shape
    if m.max() == 0:
        return None

    heights = np.zeros(W, dtype=np.int32)
    best_area = 0
    best_bbox = None

    for y in range(H):
        row = m[y]
        heights = heights + 1
        heights[row == 0] = 0

        stack: List[int] = []
        for i in range(W + 1):
            cur_h = heights[i] if i < W else 0
            while stack and cur_h < heights[stack[-1]]:
                h = heights[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                right = i
                area = h * (right - left)
                if area > best_area:
                    best_area = area
                    x1, x2 = left, right
                    y2 = y + 1
                    y1 = y2 - h
                    best_bbox = (x1, y1, x2, y2)
            stack.append(i)

    return best_bbox

def _iter_images(root: Union[str, Path]) -> List[str]:
    root = str(root)
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(dirpath, fn))
    return paths

def _safe_xyxy_from_obj(obj: Any) -> Optional[Tuple[float, float, float, float]]:
    """
    Accepts:
      - tuple/list (x1,y1,x2,y2)
      - dict with keys xmin,ymin,xmax,ymax OR x1,y1,x2,y2
      - object with .xyxy property (your BoundingBox has this)
      - object with .box having .xyxy
    Returns None if cannot parse.
    """
    if obj is None:
        return None

    # tuple/list
    if isinstance(obj, (list, tuple)) and len(obj) == 4:
        x1, y1, x2, y2 = obj
        return float(x1), float(y1), float(x2), float(y2)

    # dict
    if isinstance(obj, dict):
        if all(k in obj for k in ("xmin", "ymin", "xmax", "ymax")):
            return float(obj["xmin"]), float(obj["ymin"]), float(obj["xmax"]), float(obj["ymax"])
        if all(k in obj for k in ("x1", "y1", "x2", "y2")):
            return float(obj["x1"]), float(obj["y1"]), float(obj["x2"]), float(obj["y2"])

    # has .xyxy
    if hasattr(obj, "xyxy"):
        xyxy = getattr(obj, "xyxy")
        if isinstance(xyxy, (list, tuple)) and len(xyxy) == 4:
            return float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

    # has .box.xyxy (DetectionResult.box)
    if hasattr(obj, "box") and hasattr(obj.box, "xyxy"):
        xyxy = obj.box.xyxy
        if isinstance(xyxy, (list, tuple)) and len(xyxy) == 4:
            return float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

    return None


# -------------------------
# Config
# -------------------------

@dataclass
class CropConfig:
    # new: "grounder" family modes
    mode: str  # "center" | "bbox" | "segmentation" | "grounder_mask" | "grounder_box" | "grounder_mask_or_box"
    out_size: int = 224

    # bbox padding when cropping from boxes
    bbox_padding: float = 0.30

    # if you want to avoid tiny detections
    min_box_side: int = 10

    # if true: for images with multiple detections, save one crop per detection
    multi_crop: bool = True

    # if false and multi_crop: keep only the biggest crop (by box area)
    keep_biggest_only: bool = False


# -------------------------
# Core crop from grounder outputs
# -------------------------

def _box_area(xyxy: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    # (1) try max-rectangle-inside-mask (tight but conservative)
    bb = find_biggest_bbox_inside_mask(mask)
    if bb is not None:
        return bb

    # (2) fallback to plain bounding box of non-zero pixels
    ys, xs = np.where(mask.astype(np.uint8) > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2, y2)

def _crop_from_detection(
    img: Image.Image,
    cfg: CropConfig,
    det: Dict[str, Any],
) -> Optional[Image.Image]:
    """
    det can contain:
      - "mask": np.ndarray (H,W)
      - "box": tuple/list (x1,y1,x2,y2) OR dict xmin/ymin/xmax/ymax OR a BoundingBox-like obj
    """
    mode = cfg.mode.lower()

    mask = det.get("mask")
    box_obj = det.get("box")
    xyxy = _safe_xyxy_from_obj(box_obj)

    # choose crop source
    use_mask = mode in {"segmentation", "grounder_mask", "grounder_mask_or_box"} and mask is not None
    use_box  = mode in {"bbox", "grounder_box", "grounder_mask_or_box"} and xyxy is not None

    # if mask chosen, convert to bbox
    if use_mask:
        mbb = _mask_bbox(mask)
        if mbb is not None:
            xyxy = (float(mbb[0]), float(mbb[1]), float(mbb[2]), float(mbb[3]))
            use_box = True  # now crop via bbox_crop

    if not use_box or xyxy is None:
        return None

    x1, y1, x2, y2 = xyxy
    if (x2 - x1) < cfg.min_box_side or (y2 - y1) < cfg.min_box_side:
        return None

    return bbox_crop(img, cfg.out_size, bbox_xyxy=xyxy, padding=cfg.bbox_padding)


def _normalize_grounder_inputs(
    grounder_outputs: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Accepts any of:
      A) masks_by_relpath: rel -> [mask, mask, ...]
      B) boxes_by_relpath: rel -> [xyxy, xyxy, ...] (xyxy can be tuple/list/dict/BoundingBox)
      C) detections_by_relpath: rel -> [DetectionResult, ...] (from your GrounderSAM2)
      D) dict rel -> list of dicts already: {"mask":..., "box":...}

    Returns:
      rel -> list of {"mask": Optional[np.ndarray], "box": Optional[Any]}
    """
    if grounder_outputs is None:
        return {}

    out: Dict[str, List[Dict[str, Any]]] = {}

    # rel -> list[dict] already?
    if isinstance(grounder_outputs, dict):
        for rel, items in grounder_outputs.items():
            if not isinstance(items, list):
                continue

            norm_list: List[Dict[str, Any]] = []

            # case A: list of np.ndarray masks
            if items and isinstance(items[0], np.ndarray):
                for m in items:
                    norm_list.append({"mask": m, "box": None})
                out[rel] = norm_list
                continue

            # case B: list of boxes (tuple/list/dict/BoundingBox)
            if items and (isinstance(items[0], (list, tuple, dict)) or hasattr(items[0], "xyxy")):
                for b in items:
                    # could also be dict with mask+box
                    if isinstance(b, dict) and ("mask" in b or "box" in b):
                        norm_list.append({"mask": b.get("mask"), "box": b.get("box")})
                    else:
                        norm_list.append({"mask": None, "box": b})
                out[rel] = norm_list
                continue

            # case C: list of DetectionResult-like objects
            if items and hasattr(items[0], "box"):
                for d in items:
                    norm_list.append({"mask": getattr(d, "mask", None), "box": getattr(d, "box", None)})
                out[rel] = norm_list
                continue

            # default: try dict-ish
            for it in items:
                if isinstance(it, dict):
                    norm_list.append({"mask": it.get("mask"), "box": it.get("box")})
            out[rel] = norm_list

    return out


# -------------------------
# Folder processing
# -------------------------

def process_folder(
    in_root: Union[str, Path],
    out_root: Union[str, Path],
    cfg: CropConfig,
    grounder_outputs: Optional[Any] = None,
    bbox_json: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """
    New behavior:
      - If cfg.mode is one of the "grounder_*" or "segmentation"/"bbox", this can use:
          * grounder_outputs (preferred): rel -> detections (masks/boxes)
          * bbox_json (legacy): rel -> [x1,y1,x2,y2]
      - If multiple detections exist:
          * if cfg.multi_crop: writes out *_det{i}.ext
          * else: chooses biggest detection (by area) and writes single file

    Notes:
      - out_root mirrors in_root folder structure.
      - rel paths must match the same rel paths used by grounder stage:
          rel = os.path.relpath(img_path, in_root)
    """
    in_root = Path(in_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # legacy bbox_json support
    bbox_map: Dict[str, List[float]] = {}
    if bbox_json:
        with open(bbox_json, "r", encoding="utf-8") as f:
            bbox_map = json.load(f)

    dets_by_rel = _normalize_grounder_inputs(grounder_outputs)

    img_paths = _iter_images(in_root)

    for p in img_paths:
        rel = os.path.relpath(p, in_root)
        out_path = str(out_root / rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = Image.open(p).convert("RGB")
        mode = cfg.mode.lower()

        # center fallback always available
        if mode == "center":
            center_crop(img, cfg.out_size).save(out_path)
            continue

        # if bbox_json provided and mode is bbox (legacy)
        if mode == "bbox" and bbox_json:
            if rel in bbox_map:
                bb = bbox_map[rel]
                bbox = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                bbox_crop(img, cfg.out_size, bbox, padding=cfg.bbox_padding).save(out_path)
            else:
                center_crop(img, cfg.out_size).save(out_path)
            continue

        # grounder-driven modes (mask/box/or both)
        if mode in {"segmentation", "grounder_mask", "grounder_box", "grounder_mask_or_box", "bbox"}:
            det_list = dets_by_rel.get(rel, [])

            # For plain "bbox" without bbox_json, allow grounder boxes if provided
            if not det_list:
                # no detections => fallback
                center_crop(img, cfg.out_size).save(out_path)
                continue

            # build crops
            crops: List[Tuple[float, Image.Image]] = []  # (area, crop)
            for det in det_list:
                crop = _crop_from_detection(img, cfg, det)
                if crop is None:
                    continue

                # compute area from chosen box (mask->bbox or box)
                xyxy = _safe_xyxy_from_obj(det.get("box"))
                if det.get("mask") is not None and mode in {"segmentation", "grounder_mask", "grounder_mask_or_box"}:
                    mbb = _mask_bbox(det["mask"])
                    if mbb is not None:
                        xyxy = (float(mbb[0]), float(mbb[1]), float(mbb[2]), float(mbb[3]))
                area = _box_area(xyxy) if xyxy is not None else 0.0
                crops.append((area, crop))

            if not crops:
                center_crop(img, cfg.out_size).save(out_path)
                continue

            # keep biggest only if requested
            if (not cfg.multi_crop) or cfg.keep_biggest_only:
                crops.sort(key=lambda x: x[0], reverse=True)
                crops[0][1].save(out_path)
                continue

            # multi-crop: write one per detection
            for i, (_, cimg) in enumerate(crops):
                out_i = _with_det_suffix(out_path, i)
                os.makedirs(os.path.dirname(out_i), exist_ok=True)
                cimg.save(out_i)

            continue

        raise ValueError(f"Unknown mode: {cfg.mode}")


# -------------------------
# Convenience wrapper class
# -------------------------

class Cropper:
    def __init__(
        self,
        cfg: CropConfig,
        grounder_outputs: Optional[Any] = None,
        bbox_json: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.cfg = cfg
        self.grounder_outputs = grounder_outputs
        self.bbox_json = bbox_json
        self.seed = seed

    def process_folder(self, in_root: Union[str, Path], out_root: Union[str, Path]) -> None:
        process_folder(
            in_root=in_root,
            out_root=out_root,
            cfg=self.cfg,
            grounder_outputs=self.grounder_outputs,
            bbox_json=self.bbox_json,
            seed=self.seed,
        )


# -------------------------
# Example usage with GrounderSAM2
# -------------------------
# Assuming you added GrounderSAM2.run_on_images_in_dir(..., return_detections=True)
#
# masks_by_rel, dets_by_rel = grounderSAM.run_on_images_in_dir(
#     in_root=concept_img_dir,
#     labels=labels,
#     polygon_refinement=True,
#     return_detections=True,
# )
#
# # Option A: crop from masks (mask->bbox)
# cropper = Cropper(CropConfig(mode="grounder_mask_or_box", out_size=224, bbox_padding=0.2),
#                   grounder_outputs=dets_by_rel)
# cropper.process_folder(concept_img_dir, crops_dir)
#
# # Option B: if you only have masks_by_rel
# cropper = Cropper(CropConfig(mode="grounder_mask", out_size=224, bbox_padding=0.0),
#                   grounder_outputs=masks_by_rel)
# cropper.process_folder(concept_img_dir, crops_dir)
#
# # Option C: if you only have boxes_by_rel (rel -> [xyxy,...])
# cropper = Cropper(CropConfig(mode="grounder_box", out_size=224, bbox_padding=0.3),
#                   grounder_outputs=boxes_by_rel)
# cropper.process_folder(concept_img_dir, crops_dir)
