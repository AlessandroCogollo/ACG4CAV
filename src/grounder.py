from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from cropper import crop_image, CropConfig, center_crop

SUPPORTED_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_image_files(images_dir: Path, exts: Set[str] = SUPPORTED_EXTS) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"images_dir is not a directory: {images_dir}")

    files = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def choose_best_box(results) -> Optional[Tuple[float, float, float, float]]:
    """Return (x1,y1,x2,y2) for highest-score box, or None if no boxes."""
    boxes = results.get("boxes")
    scores = results.get("scores")
    if boxes is None or scores is None or boxes.numel() == 0:
        return None
    idx = int(scores.argmax().item())
    x1, y1, x2, y2 = boxes[idx].tolist()
    print(f"Best box score: {float(scores[idx]):.3f} | box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
    return float(x1), float(y1), float(x2), float(y2)


@dataclass
class GrounderConfig:
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    cache_dir: Path = Path("../models/hf_cache")
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    default_text_query: str = "zebra"


class Grounder:
    def __init__(
        self,
        cfg: GrounderConfig = GrounderConfig(),
        device: Optional[str] = None,
    ):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_id, cache_dir=str(self.cfg.cache_dir)
        )
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            self.cfg.model_id, cache_dir=str(self.cfg.cache_dir)
        ).to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image, text_query: Optional[str] = None):
        """Run GroundingDINO on a PIL image and return post-processed results dict."""
        tq = text_query or self.cfg.default_text_query

        inputs = self.processor(images=image, text=tq, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.cfg.box_threshold,
            text_threshold=self.cfg.text_threshold,
            target_sizes=[image.size[::-1]],  # (h, w)
        )[0]
        return results

    def run_on_images_in_dir(
        self,
        images_dir: Path,
        text_query: Optional[str] = None,
        verbose: bool = True,
    ):
        image_files = iter_image_files(images_dir)
        if not image_files:
            print(f"No supported images found in: {images_dir}")
            return

        if verbose:
            print(f"Found {len(image_files)} image(s) in {images_dir} | device={self.device}")

        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[SKIP] {img_path} | failed to open: {e}")
                continue

            results = self.predict(image, text_query=text_query)

            if verbose:
                print(f"\n=== {img_path.name} ===")
                print("Boxes:", int(results["boxes"].shape[0]))

                for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                    x1, y1, x2, y2 = box.tolist()
                    print(f"{label} | score={float(score):.3f} | box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

    # Optional: since you already import cropper utils, here’s a practical helper.
    def save_best_crop(
        self,
        img_path: Path,
        out_path: Path,
        text_query: Optional[str] = None,
        out_size: int = 256,
        bbox_padding: float = 0.10,
        fallback_center: bool = True,
    ) -> bool:
        """Detect best box on one image and save the cropped result. Returns True if bbox used."""
        image = Image.open(img_path).convert("RGB")
        results = self.predict(image, text_query=text_query)
        best_box = choose_best_box(results)

        if best_box is None:
            if not fallback_center:
                return False
            cropped = center_crop(image, out_size)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(out_path)
            return False

        crop_cfg = CropConfig(mode="bbox", out_size=out_size, bbox_padding=bbox_padding)
        cropped = crop_image(image, crop_cfg, bbox_xyxy=best_box)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(out_path)
        return True
