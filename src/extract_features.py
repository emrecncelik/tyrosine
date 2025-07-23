import os
import argparse
from loguru import logger
from net2brain.feature_extraction import FeatureExtractor
from net2brain.utils.download_datasets import DatasetNSD_872
from config import EXPERIMENT_MODALITY, EXPERIMENT_TASK

# DOES NOT WORK YET


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
    dataset = DatasetNSD_872(data_dir=args.data_dir)
    dataset = dataset.load_dataset("/media/stuff")
    image_paths = dataset["NSD_872_images"]
    text_paths = dataset["NSD_872_captions"]

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

            feature_extractor = FeatureExtractor(model=model.name, netset=model.netset)

            logger.info(f"Extracting features for modality: {modality}")
            feature_extractor.extract(
                data_path=data_path,
                save_path=f"{config['feature_directory']}/{model.feature_directory}/{modality}",
                consolidate_per_layer=False,
            )
