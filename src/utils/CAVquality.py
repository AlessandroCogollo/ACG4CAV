# TODO: build this
import random
from os import makedirs
from pathlib import Path

import yaml
from sympy.printing.pytorch import torch

from CAVtrainer import CAVtrainer, CAVConfig
from utils.activations import extract_activations
from utils.modelloader import load_hf_timm_model_from_folder

RANDOM_SEEDS = [42, 123, 999, 2021, 7]
CHOSEN_LAYERS = []

base_path = Path(__file__).parent.parent

with open(Path.joinpath(base_path, "configs", "dev.yaml"), 'r') as configs_file:
	configs = yaml.safe_load(configs_file)

model_path = Path.joinpath(base_path, "data/models")
model_name = configs["data"]["base_model"]

if not Path.joinpath(model_path, model_name).exists():
	raise FileNotFoundError(f"Model folder not found: {Path.joinpath(model_path, model_name)}")

makedirs(Path.joinpath(model_path, model_name), exist_ok=True)
model = load_hf_timm_model_from_folder(str(Path.joinpath(model_path, model_name)), model_name, device=None)

device = "cuda" if torch.cuda.is_available() else "cpu"

chosen_concept = configs["concepts"]["concepts"][0]["name"]  # Assuming at least one concept exists

for seed in RANDOM_SEEDS:
	random.seed(seed)

	# TESTING EVOLUTION & STABILITY OF CAV IN DIFFERENT LAYERS
	cav_list = []
	for layer in CHOSEN_LAYERS:

		cavtrainer = CAVtrainer(model, Path.joinpath(base_path,"runs"))

		concept_acts = extract_activations(
			model,
			Path.joinpath(base_path, "data", "crops", chosen_concept),
			layer,
			device
		)

		random_acts = extract_activations(
			model,
			Path.joinpath(base_path, "data", "negatives"),
			layer,
			device
		)

		cav_auto = cavtrainer.train_cav(concept_acts, random_acts, CAVConfig(concept_name=chosen_concept, layer_name=layer, output_path=str(Path.joinpath(base_path,"runs"))))
		cav_list.append(cav_auto)

	# TODO: compute similarity between cavs in cav_list to evaluate stability across layers
	# print(f"Similarity between CAVs for layer {layer}: {similarity_metric(cav_list)}")

# TODO: implement bootstrap negative sampling to evaluate stability of CAVs across different random samples of negatives
