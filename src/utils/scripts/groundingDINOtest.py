import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-base"
IMAGE_PATH = "data/images/dev/test.jpg"
TEXT_QUERY = "wheel"

def main():
    device = "cpu"
    image = Image.open(IMAGE_PATH).convert("RGB")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()

    inputs = processor(images=image, text=TEXT_QUERY, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.30,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]],  # (h, w)
    )[0]

    print("Boxes:", results["boxes"].shape[0])
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = box.tolist()
        print(f"{label} | score={score:.3f} | box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

if __name__ == "__main__":
    main()
