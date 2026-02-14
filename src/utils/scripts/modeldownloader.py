# THANKS CHATGPT FOR PROVIDING THIS CODE!

from __future__ import annotations

import json
from os import makedirs, path

import torch
import timm


def _ensure_dir(p: str) -> None:
    makedirs(p, exist_ok=True)


def _write_lines(fp: str, lines: list[str]) -> None:
    with open(fp, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def _imagenet_class_names() -> list[str]:
    """
    Uses timm's ImageNetInfo metadata when available to produce 1000 human-readable class names.
    Fallback: returns synset IDs if detailed names aren't available in your timm version.
    """
    try:
        from timm.data.imagenet_info import ImageNetInfo  # timm>=0.8-ish
        info = ImageNetInfo()
        # Prefer detailed labels if present; otherwise synsets
        if hasattr(info, "label_names") and info.label_names is not None:
            names = list(info.label_names())
            if len(names) >= 1000:
                return names[:1000]
        if hasattr(info, "synsets") and info.synsets is not None:
            syn = list(info.synsets())
            if len(syn) >= 1000:
                return syn[:1000]
    except Exception:
        pass

    # Last-resort fallback (won't be pretty, but keeps the pipeline running)
    return [f"class_{i}" for i in range(1000)]


def download_and_save_model(
    hf_repo_id: str,
    out_dir: str,
    out_stem: str,
) -> None:
    """
    Downloads a pretrained ImageNet classifier from HF Hub via timm and saves it locally.
    """
    _ensure_dir(out_dir)

    # Download + instantiate from Hugging Face Hub
    model = timm.create_model(f"hf-hub:{hf_repo_id}", pretrained=True)
    model.eval()

    # Save weights
    weights_path = path.join(out_dir, f"{out_stem}-weights.pth")
    torch.save(model.state_dict(), weights_path)

    # Save provenance (which HF repo was used)
    meta_path = path.join(out_dir, f"{out_stem}-hf_source.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"hf_repo_id": hf_repo_id}, f, indent=2)

    # Save ImageNet class labels
    classes_path = path.join(out_dir, f"{out_stem}-imagenet-classes.txt")
    _write_lines(classes_path, _imagenet_class_names())


if __name__ == "__main__":
    dir_path = path.dirname(path.realpath(__file__))
    base_path = path.abspath(path.join(dir_path, "../../../"))

    models_path = path.join(base_path, "data/models")

    model_resnet_path = path.join(models_path, "ResNet50V2")

    print("Creating folders...", end=" ")
    _ensure_dir(models_path)
    _ensure_dir(model_resnet_path)
    print("Done!")

    print("Downloading models from Hugging Face Hub (via timm)...", end=" ")

    # ResNet50V2: timm/resnetv2_50.a1h_in1k :contentReference[oaicite:4]{index=4}
    download_and_save_model(
        hf_repo_id="timm/resnetv2_50.a1h_in1k",
        out_dir=model_resnet_path,
        out_stem="ResNet50V2",
    )

    print("Done!")
