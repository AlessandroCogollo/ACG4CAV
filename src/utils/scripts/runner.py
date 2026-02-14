from os import makedirs
from pathlib import Path

import torch
import yaml

from CAVtrainer import save_cav, compute_cav
from filterer import Filterer, FiltererConfig
from similarities import Similarities, SimilaritiesConfig
from utils.activations import extract_activations
from utils.modelloader import load_hf_timm_model_from_folder

device = "cuda" if torch.cuda.is_available() else "cpu"
base_path = Path(__file__).resolve().parents[3]
model_path = Path.joinpath(base_path, "data/models")

def main():
	# LOADING CONFIG FILES
	device = "cuda" if torch.cuda.is_available() else "cpu"

	print("Reading Config File...\n")
	with open(Path.joinpath(base_path, "configs", "dev.yaml"), 'r') as configs_file:
		configs = yaml.safe_load(configs_file)

		grounder_hf_id = configs["grounding"]["hf_model_id"]
		cache_dir = Path.joinpath(base_path, configs["data"]["cache_dir"])
		img_dir = Path.joinpath(base_path, configs["data"]["images_dir"])
		crops_dir = Path.joinpath(base_path, configs["data"]["crops_dir"])
		model_name = configs["data"]["base_model"]
		with open(Path.joinpath(base_path, configs["data"]["concepts_file"]), 'r') as concepts_file:
			concepts = yaml.safe_load(concepts_file)

	print("Initializing Base Model")

	#TODO: manage error if model not found, or if metadata file missing, etc.
	makedirs(Path.joinpath(model_path, model_name), exist_ok=True)
	model = load_hf_timm_model_from_folder(str(Path.joinpath(model_path, model_name)), model_name, device=None)

	# Getting Model's layers names
	# for name, _ in model.named_modules():
	#	print(name)

	print("Initializing Grounder...\n")
	# grounder = Grounder(GrounderConfig(model_id=grounder_hf_id, cache_dir=cache_dir), device=device)

	print("Reading concepts")
	for concept in concepts["concepts"]:
		# bboxes = grounder.run_on_images_in_dir(Path.joinpath(img_dir, concept["name"]), concept["name"])

		if configs["cropping"]["enabled"] is True:
			print("Cropping images...")
			cropper = Cropper(CropConfig(mode=configs["cropping"]["method"], out_size=224), bbox_json=bboxes, saliency_root=None, seed=configs["run"]["seed"])
			print(str(Path.joinpath(img_dir, concept["name"])), str(Path.joinpath(img_dir, "crops", concept["name"])))
			cropper.process_folder(str(Path.joinpath(img_dir, concept["name"])), str(Path.joinpath(img_dir, "crops", concept["name"])))

		if configs["filtering"]["enabled"] is True:
			print("Filtering images...")
			filterer = Filterer(FiltererConfig(method=configs["filtering"]["method"]))
			filterer.process_folder(str(Path.joinpath(img_dir, "crops", concept["name"])), str(Path.joinpath(img_dir, "filtered", concept["name"])))

		layer_name = concept["data"]["layer_name"]

		print("Training CAVs...")

		# --- Compute CAVs for generated images ---

		concept_acts = extract_activations(
			model,
			Path.joinpath(base_path, "data", "crops", concept["name"]),
			layer_name,
			device
		)

		random_acts = extract_activations(
			model,
			Path.joinpath(base_path, "data", "negatives"),
			layer_name,
			device
		)

		cav_vector = compute_cav(concept_acts, random_acts)

		save_cav(
			cav_vector,
			concept["name"],
			layer_name,
			Path.joinpath(base_path,"runs")
		)

		# --- Extract CAVs for ground truth images ---
		# --------------------------------------------

		print("Computing similarity scores...")
		similarity_computer = Similarities(SimilaritiesConfig())
		similarity_scores = similarity_computer.compute_similarity_scores(cav_vector, cav_vector)

if __name__ == "__main__":
	main()
