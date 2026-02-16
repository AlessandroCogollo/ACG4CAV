import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def extract_activations(
    model,
    image_dir,
    layer_name,
    device,
    batch_size=32,
    num_workers=4
):
    model.eval()
    model.to(device)

    activations = []

    # ImageNet normalization (IMPORTANT)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ImageFolderDataset(image_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # --- Hook ---
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    modules = dict(model.named_modules())

    if layer_name not in modules:
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    hook = modules[layer_name].register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            _ = model(batch)

    hook.remove()

    activations = torch.cat(activations, dim=0)
    activations = activations.view(activations.size(0), -1)

    return activations.numpy()
