import glob
import numpy as np
from tqdm import tqdm


def feature_postprocess(features: np.ndarray, op: str) -> np.ndarray:
    features = np.squeeze(features)

    if op == "mean":
        features = features.mean(axis=0, keepdims=True)
    elif op == "cls":  # assuming CLS token is the first token
        features = features[0, :]
        features = features[np.newaxis, :]
    elif op == "eos":  # assuming EOS token is the last token
        features = features[-1, :]
        features = features[np.newaxis, :]
    elif op == "flatten":
        features = features.flatten()
        features = features[np.newaxis, :]
    else:
        raise ValueError(
            f"Unsupported operation: {op}. Supported operations are: 'mean', 'cls', 'eos', 'flatten'."
        )

    return features


def process_and_update_features(feature_dir: str, op: str = "mean") -> None:
    """
    Process and update features in the specified directory, saving processed features in the same file.

    Args:
        feature_dir (str): Directory containing feature files.
        op (str): Operation to apply on features. Default is 'mean'.

    Usage:
        >>> from src.utils import process_and_update_features
        >>> from net2brain.utils.download_datasets import DatasetNSD_872
        >>> from net2brain.feature_extraction import FeatureExtractor
        >>>
        >>> nsd_dataset = DatasetNSD_872()
        >>> paths = nsd_dataset.load_dataset()
        >>>
        >>> extractor = FeatureExtractor(
        ...     "bert-base-uncased",
        ...     netset="Huggingface",
        ...     device="cpu",
        ... )
        >>>
        >>> extractor.extract(
        ...     paths["NSD_872_captions"],
        ...     save_path="./bert_features",
        ...     consolidate_per_layer=False,
        ...     layers_to_extract=["encoder.layer." + str(i) for i in range(12)],
        ... )
        >>>
        >>> process_and_update_features("bert_features", op='mean')
    """
    feature_files = glob.glob(f"{feature_dir}/*.npz")
    for file in tqdm(feature_files, desc=f"Processing feature files with op={op}"):
        data = np.load(file, allow_pickle=True)
        processed = {}
        for k in data.keys():
            features = data[k]
            processed_features = feature_postprocess(features, op=op)
            processed[k] = processed_features
        np.savez(file, **processed)
