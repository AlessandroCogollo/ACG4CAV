from os import makedirs
from pathlib import Path

import numpy as np
import torch
import yaml

from CAVtrainer import CAVtrainer, CAVConfig
from cropper import Cropper, CropConfig
from filterer import Filterer, FiltererConfig
from grounderSAM2 import GrounderSAM2, GrounderSAM2Config
from similarities import Similarities, SimilaritiesConfig
from utils.activations import extract_activations
from utils.logger.logger import ExperimentLogger
from utils.modelloader import load_hf_timm_model_from_folder

device = "cuda" if torch.cuda.is_available() else "cpu"
base_path = Path(__file__).resolve().parents[3]
model_path = Path.joinpath(base_path, "data/models")

def main():
    # LOADING CONFIG FILES
    print("Reading Config File...")
    with open(Path.joinpath(base_path, "configs", "dev.yaml"), 'r') as configs_file:
        configs = yaml.safe_load(configs_file)

        grounder_hf_id = configs["grounding"]["hf_model_id"]
        sam_hf_id = configs["sam"]["hf_model_id"]
        cache_dir = Path.joinpath(base_path, configs["data"]["cache_dir"])
        img_dir = Path.joinpath(base_path, configs["data"]["images_dir"])
        crops_dir = Path.joinpath(base_path, configs["data"]["crops_dir"])
        concept_dir = Path.joinpath(base_path, "data", "images")
        negatives = Path.joinpath(base_path, "data", "negatives")
        model_name = configs["data"]["base_model"]

        # Set fixed seeds for reproducibility

        seed = configs["run"]["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        # --------------------------------

        runs_root = Path.joinpath(base_path, "runs")
        runs_root.mkdir(exist_ok=True)

        logger = ExperimentLogger(runs_root)
        logger.save_config(Path.joinpath(base_path, "configs", "dev.yaml"))

        with open(Path.joinpath(base_path, configs["data"]["concepts_file"]), 'r') as concepts_file:
            concepts = yaml.safe_load(concepts_file)

    print("Initializing Base Model...")

    makedirs(Path.joinpath(model_path, model_name), exist_ok=True)
    model = load_hf_timm_model_from_folder(str(Path.joinpath(model_path, model_name)), model_name, device=None)

    # Getting Model's layers names
    # for name, _ in model.named_modules():
    #	print(name)

    # initializing GroundingSAM

    print("Initializing Grounding SAM...")

    grounderSAM = GrounderSAM2(GrounderSAM2Config(
        grounder_id=grounder_hf_id,
        segmenter_id=sam_hf_id,
        grounder_box_threshold=configs["grounding"]["box_threshold"],
        grounder_text_threshold=configs["grounding"]["text_threshold"],
        device=device
    ))

    # -------------------------

    print("Reading concepts...")
    for concept in concepts["concepts"]:
        concept_img_dir = Path.joinpath(img_dir, concept["name"])
        labels = concept["prompt"]

        # using GrounderSAM to extract the segmentation info
        masks_by_rel = grounderSAM.run_on_images_in_dir(
			in_root=concept_img_dir,
			labels=labels,
			polygon_refinement=True,
		)

        # -----------------------------------------------

        if configs["cropping"]["enabled"] is True:
            print("Cropping images...")
            cropper = Cropper(
                CropConfig(mode=configs["cropping"]["method"], out_size=224),
                masks_by_rel
            )
            cropper.process_folder(
                str(concept_img_dir),
                str(Path.joinpath(crops_dir, concept["name"])),
            )

        if configs["filtering"]["enabled"] is True:
            print("Filtering images...")
            filterer = Filterer(FiltererConfig(method=configs["filtering"]["method"]))
            filterer.process_folder(str(Path.joinpath(crops_dir,concept["name"])), str(Path.joinpath(img_dir, "filtered", concept["name"])))

        dataset_stats = {
            "n_concept_images": len([x for x in Path.iterdir(Path.joinpath(concept_dir, concept["name"])) if x.is_file()]),
            "n_random_images": len([x for x in Path.iterdir(negatives) if x.is_file()]),
        }

        logger.log_dataset_stats(concept["name"], dataset_stats)

        layer_name = configs["data"]["layer_name"]

        print("Training CAVs...")

        # --- Compute CAVs for generated images ---
        cav_trainer = CAVtrainer(model, Path.joinpath(base_path,"runs"))

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

        # --- Extract CAVs for ground truth images ---

        ground_truth_acts = extract_activations(
            model,
            Path.joinpath(base_path, "data", "groundtruth", concept["name"]),
            layer_name,
            device
        )

        # --------------------------------------------

        cav_groundtruth = cav_trainer.train_cav(ground_truth_acts, random_acts, CAVConfig(concept_name=concept["name"], layer_name=layer_name, output_path=str(Path.joinpath(base_path,"runs"))))
        cav_auto = cav_trainer.train_cav(concept_acts, random_acts, CAVConfig(concept_name=concept["name"], layer_name=layer_name, output_path=str(Path.joinpath(base_path,"runs"))))

        print("Computing similarity scores...")
        similarity_computer = Similarities(SimilaritiesConfig(method=configs["similarity"]["method"]))
        similarity_scores = similarity_computer.compute_similarity_scores(cav_groundtruth, cav_auto)

        print(similarity_scores)

        # Logging results -----------------------------

        logger.save_numpy(cav_auto, f"{concept['name']}_auto_cav.npy")
        logger.save_numpy(cav_groundtruth, f"{concept['name']}_gt_cav.npy")

        metrics = {
            "cosine_similarity": float(similarity_scores),
            "n_concept_samples": int(len(concept_acts)),
            "n_random_samples": int(len(random_acts))
        }

        logger.log_metrics(concept["name"], metrics)

        # --------------------------------------------

if __name__ == "__main__":
    main()
