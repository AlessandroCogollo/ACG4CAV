import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
IMAGE_PATH = r"C:\Users\cogol\PycharmProjects\ACG4CAV\data\images\zebra.jpg"
CACHE_DIR = "../../../models/hf_cache"  # persistent local folder
TEXT_QUERY = "zebra"

def main():
    device = "cpu"
    image = Image.open(IMAGE_PATH).convert("RGB")

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR).to(device)
    model.eval()

    inputs = processor(images=image, text=TEXT_QUERY, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.30,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]],  # (h, w)
    )[0]

    print("Boxes:", results["boxes"].shape[0])
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = box.tolist()
        print(f"{label} | score={score:.3f} | box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

        # --- TESTING CROPPER --- #
        from utils.cropper import crop_image, CropConfig
        crop_image(image, CropConfig(mode="bbox", out_size=256), bbox_xyxy=(x1, y1, x2, y2)).show()

if __name__ == "__main__":
    main()
