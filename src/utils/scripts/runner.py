import torch
import yaml

from pathlib import Path
from grounder import Grounder, GrounderConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
base_path = Path(__file__).resolve().parents[3]

def main():
	# LOADING CONFIG FILES

	with open(Path.joinpath(base_path, "configs", "dev.yaml"), 'r') as configs_file:
		configs = yaml.safe_load(configs_file)

		grounder_hf_id = configs["grounding"]["hf_model_id"]
		cache_dir = Path.joinpath(base_path, configs["data"]["cache_dir"])
		img_dir = Path.joinpath(base_path, configs["data"]["images_dir"])
		with open(Path.joinpath(base_path, configs["data"]["concepts_file"]), 'r') as concepts_file:
			concepts = yaml.safe_load(concepts_file)


	device = "cuda" if torch.cuda.is_available() else "cpu"
	grounder = Grounder(GrounderConfig(model_id=grounder_hf_id, cache_dir=cache_dir), device=device)
	for concept in concepts:
		print(concept)
		# bboxes = grounder.run_on_images_in_dir(img_dir, concept)
		# TODO: call cropper
		# TODO: call filterer
		# TODO: call CAV trainer
		# TODO: call compute similarity scores

if __name__ == "__main__":
	main()
