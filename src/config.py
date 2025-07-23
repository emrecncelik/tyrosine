from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    name: str
    netset: str
    modality: list[str]
    extraction_layers: dict[str, list[str]]
    save_dir: Optional[str] = None

    def __post_init__(self):
        if self.save_dir is None:
            self.save_dir = f"{self.name.replace('/', '_')}"

    def save_dir_for_modality(self, modality: str) -> str:
        return f"{self.save_dir}/{modality}"


EXPERIMENT_MODALITY = {
    "models": [
        ModelConfig(
            name="bert-base-uncased",
            netset="Huggingface",
            modality=["language"],
            extraction_layers={
                "language": ["encoder.layer." + str(i) for i in range(12)]
            },
        ),
        ModelConfig(
            name="vit_large_patch16_224_in21k",  # 224x224, 16x16 patches, imagenet
            netset="Timm",
            modality=["vision"],
            extraction_layers={"vision": ["blocks." + str(i) for i in range(24)]},
        ),
        ModelConfig(
            name="ViT-L_-_14",
            netset="Clip",
            modality=["vision", "language"],
            extraction_layers={
                "vision": ["visual.transformer.resblocks." + str(i) for i in range(24)],
                "language": ["transformer.resblocks." + str(i) for i in range(12)],
            },
        ),
    ],
    "feature_directory": "features_modality",
    "rdm_directory": "rdms_modality",
}

EXPERIMENT_TASK = {}


FMRI_RDMS = {
    "bodies": {
        "FBA-1": "floc-bodies/subject_{}_merged_FBA-1_both.npz",
        "FBA-2": "floc-bodies/subject_{}_merged_FBA-2_both.npz",
        "EBA": "floc-bodies/subject_{}_merged_EBA_both.npz",
    },
    "faces": {
        "FFA-1": "floc-faces/subject_{}_merged_FFA-1_both.npz",
        "FFA-2": "floc-faces/subject_{}_merged_FFA-2_both.npz",
        "FOA": "floc-faces/subject_{}_merged_FOA_both.npz",
    },
    "places": {
        "OPA": "floc-places/subject_{}_merged_OPA_both.npz",
        "PPA": "floc-places/subject_{}_merged_PPA_both.npz",
        "RSC": "floc-places/subject_{}_merged_RSC_both.npz",
    },
}
