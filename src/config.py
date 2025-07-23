from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    netset: str
    modality: list[str]
    extraction_layers: dict[str, str]
    feature_directory: str = None

    def __post_init__(self):
        if self.feature_directory is None:
            self.feature_directory = f"{self.name.replace('/', '_')}"


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
            name="vit_large_patch16_224_in21k",  # 224x224, 16x16 patches
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
                "language": ["transformer.resblocks.11" + str(i) for i in range(12)],
            },
        ),
    ],
    "feature_directory": "features_modality",
    "rdm_directory": "rdms_modality",
}

EXPERIMENT_TASK = {}
