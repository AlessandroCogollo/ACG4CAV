import torch
from pathlib import Path

from grounder import Grounder, GrounderConfig

GROUNDER_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
IMAGES_DIR = Path(r"../../../data/images/zebra")
CACHE_DIR = Path("../../models/hf_cache")
TEXT_QUERY = "zebra"
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# TODO: cropper =
	grounder = Grounder(GrounderConfig(model_id=GROUNDER_MODEL_ID, cache_dir=CACHE_DIR), device=device)
	bboxes = grounder.run_on_images_in_dir(IMAGES_DIR, TEXT_QUERY)
	print(bboxes)
	# TODO: crop_image(image, CropConfig(mode="bbox", out_size=256), bbox_xyxy=(x1, y1, x2, y2)).show()


if __name__ == "__main__":
	main()
