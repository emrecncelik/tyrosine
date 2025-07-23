import glob
import math
import torch
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
        np.savez(file, **processed)  # TODO: dont update all at once to save memory


def filepath_to_nsd_id(filename: str) -> int:
    return int(filename.split("-")[-1].split(".")[0])


def filepath_to_fmri_index(filename: str) -> int:
    return int(filename.split("/")[-1].split("_")[0].split("-")[-1])


def get_fmri_indices(paths: list[str]) -> list[int]:
    return [filepath_to_fmri_index(p) for p in paths]


def filter_filepaths(paths: list[str], ids: list[int]) -> list[str]:
    return [p for p in paths if filepath_to_nsd_id(p) in ids]


def filter_common(paths: str, stimuli1000path: str = "./stimuli1000.csv"):
    ids = pd.read_csv(stimuli1000path)["nsd_id"].tolist()
    return filter_filepaths(paths=paths, ids=ids)


def to_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    # Taken from Net2Brain to convert RDMs to rsatoolbox format.
    Converts a condensed distance vector to a symmetric distance matrix.

    Parameters
    ----------
    x : torch.Tensor
        A condensed distance vector of shape (n * (n - 1) / 2, ) or (b, n * (n - 1) / 2).

    Returns
    -------
    torch.Tensor
        A symmetric distance matrix of shape (n, n) or (b, n, n).
    """
    shape = x.shape
    d = math.ceil(math.sqrt(shape[-1] * 2))

    if d * (d - 1) != shape[-1] * 2:
        raise ValueError("The input must be a condensed distance matrix.")

    i = torch.triu_indices(d, d, offset=1, device=x.device)
    if len(shape) == 1:
        out = torch.zeros(d, d, dtype=x.dtype, device=x.device)
        out[i[0], i[1]] = x
        out[i[1], i[0]] = x
    elif len(shape) == 2:
        out = torch.zeros(shape[0], d, d, dtype=x.dtype, device=x.device)
        out[..., i[0], i[1]] = x
        out[..., i[1], i[0]] = x
    else:
        raise ValueError(f"Input must be 2- or 3-d but has {len(shape)} dimension(s).")
    return out
