import os
import torch
import argparse
from loguru import logger
from net2brain.feature_extraction import FeatureExtractor
from net2brain.rdm_creation import RDMCreator
from net2brain.utils.download_datasets import DatasetNSD_872
from config import EXPERIMENT_MODALITY, EXPERIMENT_TASK
from utils import process_and_update_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run feature extraction and RDM creation."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./",
        help="Parent directory containing the dataset directory (thanks Net2Brain). Default is current directory.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["modality", "task"],
        default="modality",
        help="Choose the experiment type: 'modality' or 'task'. Default is 'modality'.",
    )

    args = parser.parse_args()

    config = EXPERIMENT_MODALITY if args.experiment == "modality" else EXPERIMENT_TASK
    dataset = DatasetNSD_872(path=args.data_dir)
    paths = dataset.load_dataset(path=args.data_dir)
    image_paths = paths["NSD_872_images"]
    text_paths = paths["NSD_872_captions"]

    for model in config["models"]:
        logger.info(f"Extracting features for model: {model.name} from {model.netset}")
        logger.info(f"Modalities: {model.modality}")
        logger.info(f"Extraction layers: {model.extraction_layers}")

        for modality in model.modality:
            if modality == "vision":
                data_path = image_paths
            elif modality == "language":
                data_path = text_paths
            else:
                raise ValueError(f"Unsupported modality: {modality}")

            device = (
                "cuda"
                if torch.cuda.is_available() and "bert" not in model.name
                else "cpu"
            )
            feature_extractor = FeatureExtractor(
                model=model.name, netset=model.netset, device=device
            )
            feature_save_dir = os.path.join(
                config["feature_directory"],
                model.save_dir_for_modality(modality),
            )

            if not os.path.exists(feature_save_dir):
                os.makedirs(feature_save_dir)
                logger.info(f"Created directory: {feature_save_dir}")

            elif os.listdir(feature_save_dir):
                logger.warning(
                    f"Directory {feature_save_dir} already exists and is not empty. Skipping extraction."
                )
                continue

            logger.info(f"Using device: {device}")
            logger.info(f"Extracting features...")

            feature_extractor.extract(
                data_path=data_path,
                save_path=feature_save_dir,
                consolidate_per_layer=False,
                layers_to_extract=model.extraction_layers[modality],
            )

            process_and_update_features(feature_save_dir, op="mean")
            logger.info(f"Features saved and processed in: {feature_save_dir}")
            del feature_extractor

            rdm_creator = RDMCreator(device=device)
            rdm_save_dir = os.path.join(
                config["rdm_directory"],
                model.save_dir_for_modality(modality),
            )

            if not os.path.exists(rdm_save_dir):
                os.makedirs(rdm_save_dir)
                logger.info(f"Created directory: {rdm_save_dir}")

            elif os.listdir(rdm_save_dir):
                logger.warning(
                    f"Directory {rdm_save_dir} already exists and is not empty. Skipping creation."
                )
                continue

            logger.info(f"Using device: {device}")
            logger.info(f"Creating RDMs...")

            rdm_creator.create_rdms(
                feature_path=feature_save_dir,
                save_path=rdm_save_dir,
                save_format="npz",
            )

            logger.info(f"RDMs saved in: {rdm_save_dir}")
            del rdm_creator

        logger.info(
            f"Finished extracting features and creating RDMs for model: {model.name}\n\n"
        )
