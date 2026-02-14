# THANKS TO CHATGPT

from __future__ import annotations

import json
from os import path
from typing import Optional

import timm
import torch


def get_device(preferred: str | None = None) -> torch.device:
	if preferred is not None:
		if preferred.startswith("cuda") and not torch.cuda.is_available():
			raise RuntimeError(
				"CUDA requested but this PyTorch build has no CUDA support."
			)
		return torch.device(preferred)

	# auto
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def load_hf_timm_model_from_folder(
	model_folder: str,
	stem: str,
	device: Optional[str] = None,
	strict: bool = True,
) -> torch.nn.Module:
	"""
	Loads a timm model instantiated from HF Hub (via timm) + your saved state_dict.

	Expected files inside `model_folder`:
	  - {stem}-hf_source.json      (contains {"hf_repo_id": "..."} )
	  - {stem}-weights.pth         (torch state_dict)

	Args:
		model_folder: e.g. ../VisualTCAV/models/InceptionV3
		stem:         e.g. "InceptionV3"
		device:       "cpu", "cuda", "cuda:0", etc. If None, auto-select.
		strict:       passed to load_state_dict.

	Returns:
		model: eval()'d model on the selected device.
	"""
	meta_path = path.join(model_folder, f"{stem}-hf_source.json")
	weights_path = path.join(model_folder, f"{stem}-weights.pth")

	if not path.exists(meta_path):
		raise FileNotFoundError(f"Missing metadata file: {meta_path}")
	if not path.exists(weights_path):
		raise FileNotFoundError(f"Missing weights file: {weights_path}")

	with open(meta_path, "r", encoding="utf-8") as f:
		meta = json.load(f)

	hf_repo_id = meta.get("hf_repo_id")
	if not hf_repo_id or not isinstance(hf_repo_id, str):
		raise ValueError(f"Invalid hf_repo_id in {meta_path}: {hf_repo_id}")


	# Create the same architecture from HF Hub
	model = timm.create_model(f"hf-hub:{hf_repo_id}", pretrained=False)

	# Load state_dict
	state = torch.load(weights_path, map_location="cpu")

	# Some people save dicts like {"state_dict": ...} — handle both
	if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
		state = state["state_dict"]

	missing, unexpected = model.load_state_dict(state, strict=strict)
	if strict and (missing or unexpected):
		# In strict=True this is usually already raised, but keep for clarity
		raise RuntimeError(f"State dict mismatch. Missing={missing}, Unexpected={unexpected}")

	device = get_device(device)
	model.to(device)
	model.eval()
	return model


def load_default_visualtcav_model(
	visual_tcav_models_dir: str,
	model_name: str,
	device: Optional[str] = None,
	strict: bool = True,
) -> torch.nn.Module:
	"""
	Convenience wrapper matching your folder layout:
	  ../VisualTCAV/models/<model_name>/<model_name>-weights.pth
	"""
	folder = path.join(visual_tcav_models_dir, model_name)
	return load_hf_timm_model_from_folder(folder, stem=model_name, device=device, strict=strict)


if __name__ == "__main__":
	# Example usage (mirrors your original path style)
	dir_path = path.dirname(path.realpath(__file__))
	visual_tcav_dir_path = path.join(dir_path, "../VisualTCAV")
	models_root = path.join(visual_tcav_dir_path, "models")

	model = load_default_visualtcav_model(models_root, "ConvNeXt", device=None, strict=True)
	x = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)
	with torch.no_grad():
		y = model(x)
	print("Output shape:", tuple(y.shape))
