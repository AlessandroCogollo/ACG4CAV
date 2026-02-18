# adaptation of https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb
# with the same structure of Cropper, Filterer, etc. for easier integration into the pipeline
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from matplotlib import pyplot as plt
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def _iter_images_dir(root: Union[str, Path]) -> List[Path]:
    root = Path(root)
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths

def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

class GrounderSAM2Config:
    def __init__(self, grounder_id: str = "IDEA-Research/grounding-dino-tiny", segmenter_id: str = "facebook/sam-vit-base", grounder_box_threshold: float = 0.3, grounder_text_threshold: float = 0.25, device: str = "cuda"):
        self.grounder_id = grounder_id
        self.segmenter_id = segmenter_id
        self.grounder_box_threshold = grounder_box_threshold
        self.grounder_text_threshold = grounder_text_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class GrounderSAM2:
    def __init__(self, config: GrounderSAM2Config):
        self.config = config
        self.cfg = config

        grounder = self.config.grounder_id if self.config.grounder_id is not None else "IDEA-Research/grounding-dino-tiny"
        self.grounder = pipeline(model=grounder, task="zero-shot-object-detection", device=self.config.device)

        segmenter = self.config.segmenter_id if self.config.segmenter_id is not None else "facebook/sam-vit-base"
        self.segmenter = AutoModelForMaskGeneration.from_pretrained(segmenter).to(self.config.device)
        self.processor = AutoProcessor.from_pretrained(segmenter)

    def detect(self, image: Image.Image, labels: List[str], threshold: float = 0.3, detector_id: Optional[str] = None ) -> List[Dict[str, Any]]:
        labels = [label if label.endswith(".") else label + "." for label in labels]
        results = self.grounder(image, candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]

        return results

    def segment(self,
            image: Image.Image,
            detection_results: List[Dict[str, Any]],
            polygon_refinement: bool = False,
    ) -> List[DetectionResult]:

        boxes = get_boxes(detection_results)
        inputs = self.processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.config.device)

        outputs = self.segmenter(**inputs)
        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results

    def grounded_segmentation(
            self,
            image: Union[Image.Image, str],
            labels: List[str],
            polygon_refinement: bool = False,
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        if isinstance(image, str):
            image = load_image(image)

        detections = self.detect(image, labels, self.config.grounder_box_threshold)
        detections = self.segment(image, detections, polygon_refinement)

        return np.array(image), detections

    def run_on_images_in_dir(
            self,
            in_root: Union[str, Path],
            labels: List[str],
            polygon_refinement: bool = False,
            return_detections: bool = False,
            save_viz_dir: Optional[Union[str, Path]] = None,
    ) -> Union[Dict[str, List[np.ndarray]], Tuple[Dict[str, List[np.ndarray]], Dict[str, List[DetectionResult]]]]:
        """
        Runs grounded segmentation on all images under in_root.

        Returns
        -------
        masks_by_relpath : dict
            rel_path -> list of masks (uint8 0/1 or 0/255 depending on refine_masks output)
        (optional) detections_by_relpath : dict
            rel_path -> list of DetectionResult
        """
        in_root = Path(in_root)

        if save_viz_dir is not None:
            save_viz_dir = Path(save_viz_dir)
            save_viz_dir.mkdir(parents=True, exist_ok=True)

        masks_by_relpath: Dict[str, List[np.ndarray]] = {}
        dets_by_relpath: Dict[str, List[DetectionResult]] = {}

        img_paths = _iter_images_dir(in_root)

        for img_path in img_paths:
            rel = os.path.relpath(img_path, in_root)

            image_array, detections = self.grounded_segmentation(
                image=str(img_path),
                labels=labels,
                polygon_refinement=polygon_refinement,
            )

            # collect masks
            masks = []
            for d in detections:
                if d.mask is not None:
                    masks.append(d.mask)
            masks_by_relpath[rel] = masks

            if return_detections:
                dets_by_relpath[rel] = detections

            if save_viz_dir is not None:
                out_path = save_viz_dir / Path(rel).with_suffix(".png")
                out_path.parent.mkdir(parents=True, exist_ok=True)

            plot_detections(image_array, detections)

        if return_detections:
            return masks_by_relpath, dets_by_relpath
        return masks_by_relpath
