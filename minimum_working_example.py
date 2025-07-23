import os
import torch
from src.config import EXPERIMENT_MODALITY
from src.utils import process_and_update_features
from net2brain.utils.download_datasets import DatasetNSD_872
from net2brain.feature_extraction import FeatureExtractor

nsd_dataset = DatasetNSD_872("/media/stuff")
paths = nsd_dataset.load_dataset("/media/stuff")


for model in EXPERIMENT_MODALITY["models"]:
    print(f"Extracting features for model: {model.name}")
    for modality in model.modality:
        print(f"Processing modality: {modality}")
        print(f"Using netset: {model.netset}")
        print(f"Extraction layers: {model.extraction_layers[modality]}")

        device = (
            "cuda" if torch.cuda.is_available() and "bert" not in model.name else "cpu"
        )
        extractor = FeatureExtractor(model.name, netset=model.netset, device=device)

        data_modality = "captions" if modality == "language" else "images"
        data_path = paths[f"NSD_872_{data_modality}"]
        output_dir = os.path.join(
            EXPERIMENT_MODALITY["feature_directory"], model.feature_directory, modality
        )
        extractor.extract(
            data_path,
            save_path=output_dir,
            consolidate_per_layer=False,
            layers_to_extract=model.extraction_layers[modality],
        )

        process_and_update_features(output_dir, op="mean")
        del extractor
        print(f"Features saved and processed in: {output_dir}")
    print(f"Finished extracting features for model: {model.name}\n\n")
