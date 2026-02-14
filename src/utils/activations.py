# THANKS TO CHATGPT-4 FOR HELPING WITH THIS CODE!

import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np

def extract_activations(model, image_dir, layer_name, device):

    activations = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Hook
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    layer = dict(model.named_modules())[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            _ = model(img)

    hook.remove()

    activations = torch.cat(activations, dim=0)
    activations = activations.view(activations.size(0), -1)

    return activations.numpy()
