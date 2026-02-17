# THANKS TO CHATGPT-4 FOR HELPING WITH THIS CODE!

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

from PIL import Image

# -------------------------
# Core crop utilities
# -------------------------

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _square_crop_box(cx: float, cy: float, side: float, w: int, h: int) -> Tuple[int, int, int, int]:
    """
    Returns a square crop box (left, top, right, bottom) clamped to image boundaries.
    If the box would go outside, it shifts it back inside.
    """
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

def random_crop(img: Image.Image, out_size: int, scale: Tuple[float, float]=(0.6, 1.0), seed: Optional[int]=None) -> Image.Image:
    """
    Random square crop.
    scale=(min,max) fraction of min(w,h) used as crop side.
    """
    import random
    if seed is not None:
        random.seed(seed)

    w, h = img.size
    base = min(w, h)
    smin, smax = scale
    smin = max(0.05, min(1.0, smin))
    smax = max(smin, min(1.0, smax))

    side = base * random.uniform(smin, smax)
    cx = random.uniform(side/2, w - side/2) if w > side else w/2
    cy = random.uniform(side/2, h - side/2) if h > side else h/2

    box = _square_crop_box(cx, cy, side, w, h)
    cropped = img.crop(box)
    return cropped.resize((out_size, out_size), resample=Image.BICUBIC)

def bbox_crop(
    img: Image.Image,
    out_size: int,
    bbox_xyxy: Tuple[float, float, float, float],
    padding: float = 0.3,
) -> Image.Image:
    """
    bbox_xyxy in pixel coordinates (x1,y1,x2,y2).
    padding restricts bbox by a fraction of max(bbox_w,bbox_h), then makes it square.
    """
    w, h = img.size
    x1, y1, x2, y2 = bbox_xyxy
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 - padding)

    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    box = _square_crop_box(cx, cy, side, w, h)
    cropped = img.crop(box)
    return cropped.resize((out_size, out_size), resample=Image.BICUBIC)

def saliency_guided_crop(
    img: Image.Image,
    out_size: int,
    saliency: Image.Image,
    crop_frac: float = 0.6,
    smoothing: int = 0,
) -> Image.Image:
    """
    Crops around saliency center-of-mass.
    - saliency: grayscale image same size as img (or will be resized).
    - crop_frac: fraction of min(w,h) used as side length.
    - smoothing: optional box blur radius (integer) to stabilize COM.
    """
    import numpy as np

    w, h = img.size
    if saliency.size != (w, h):
        saliency = saliency.resize((w, h), resample=Image.BILINEAR)

    sal = np.array(saliency.convert("L"), dtype=np.float32)

    if smoothing and smoothing > 0:
        # simple box blur
        k = smoothing
        pad = k
        sal_p = np.pad(sal, ((pad,pad),(pad,pad)), mode="edge")
        # integral image for fast box blur
        ii = sal_p.cumsum(0).cumsum(1)
        H, W = sal.shape
        out = np.zeros_like(sal)
        for y in range(H):
            y0, y1 = y, y + 2*pad + 1
            for x in range(W):
                x0, x1 = x, x + 2*pad + 1
                s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
                out[y, x] = s / ((2*pad+1)*(2*pad+1))
        sal = out

    sal = np.maximum(sal, 0.0)
    ssum = float(sal.sum())
    if ssum < 1e-6:
        # fallback
        return center_crop(img, out_size)

    ys, xs = np.indices(sal.shape)
    cx = float((sal * xs).sum() / ssum)
    cy = float((sal * ys).sum() / ssum)

    side = max(1.0, min(w, h) * float(crop_frac))
    box = _square_crop_box(cx, cy, side, w, h)
    cropped = img.crop(box)
    return cropped.resize((out_size, out_size), resample=Image.BICUBIC)

# -------------------------
# Batch folder processing
# -------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

@dataclass
class CropConfig:
    mode: str                 # "center" | "random" | "bbox" | "saliency"
    out_size: int = 224
    random_scale: Tuple[float, float] = (0.6, 1.0)
    bbox_padding: float = 0.3
    saliency_crop_frac: float = 0.6
    saliency_smoothing: int = 0

def crop_image(
    img: Image.Image,
    cfg: CropConfig,
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None,
    saliency_img: Optional[Image.Image] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    m = cfg.mode.lower()
    if m == "center":
        return center_crop(img, cfg.out_size)
    if m == "random":
        return random_crop(img, cfg.out_size, scale=cfg.random_scale, seed=seed)
    if m == "bbox":
        if bbox_xyxy is None:
            raise ValueError("bbox mode requires bbox_xyxy")
        return bbox_crop(img, cfg.out_size, bbox_xyxy=bbox_xyxy, padding=cfg.bbox_padding)
    if m == "saliency":
        if saliency_img is None:
            raise ValueError("saliency mode requires saliency_img")
        return saliency_guided_crop(
            img,
            cfg.out_size,
            saliency=saliency_img,
            crop_frac=cfg.saliency_crop_frac,
            smoothing=cfg.saliency_smoothing,
        )
    raise ValueError(f"Unknown mode: {cfg.mode}")

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
    cfg: CropConfig,
    bbox_json: Optional[str] = None,
    saliency_root: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """
    - Preserves folder structure: out_root mirrors in_root.
    - bbox_json: path to JSON mapping relative_path -> [x1,y1,x2,y2]
    - saliency_root: folder mirroring in_root containing saliency maps (same relative paths)
    """
    os.makedirs(out_root, exist_ok=True)

    bbox_map: Dict[str, List[float]] = {}
    if bbox_json:
        with open(bbox_json, "r", encoding="utf-8") as f:
            bbox_map = json.load(f)

    img_paths = _iter_images(in_root)

    for p in img_paths:
        rel = os.path.relpath(p, in_root)
        out_path = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = Image.open(p).convert("RGB")

        bbox = None
        sal = None

        if cfg.mode.lower() == "bbox":
            if rel not in bbox_map:
                # fallback center if bbox missing
                pass
                # OPTIONALLY: cropped = center_crop(img, cfg.out_size)
            else:
                bb = bbox_map[rel]
                bbox = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                cropped = crop_image(img, cfg, bbox_xyxy=bbox, seed=seed)
                cropped.save(out_path)
        elif cfg.mode.lower() == "saliency":
            if not saliency_root:
                cropped = center_crop(img, cfg.out_size)
            else:
                sal_path = os.path.join(saliency_root, rel)
                if not os.path.exists(sal_path):
                    cropped = center_crop(img, cfg.out_size)
                else:
                    sal = Image.open(sal_path)
                    cropped = crop_image(img, cfg, saliency_img=sal, seed=seed)
            cropped.save(out_path)
        else:
            cropped = crop_image(img, cfg, seed=seed)
            cropped.save(out_path)

@dataclass
class CropperConfig:
    mode: str                 # "center" | "random" | "bbox" | "saliency"
    out_size: int = 224
    random_scale: Tuple[float, float] = (0.6, 1.0)
    bbox_padding: float = 0.3
    saliency_crop_frac: float = 0.6
    saliency_smoothing: int = 0

class Cropper:
    def __init__(self, cfg: CropConfig, bbox_json: Optional[str] = None, saliency_root: Optional[str] = None, seed: Optional[int] = None):
        self.cfg = cfg
        self.bbox_json = bbox_json
        self.saliency_root = saliency_root
        self.seed = seed

    def process_folder(self, in_root: str, out_root: str) -> None:
        process_folder(
            in_root=in_root,
            out_root=out_root,
            cfg=self.cfg,
            bbox_json=self.bbox_json,
            saliency_root=self.saliency_root,
            seed=self.seed,
        )

def main():
    cfg = CropConfig(mode="center", out_size=224)

    process_folder(
        in_root="../images/zebra",
        out_root="../crops",
        cfg=cfg
    )

if __name__ == "__main__":
    main()
