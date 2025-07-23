import os
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from net2brain.utils.download_datasets import DatasetNSD_872
from sklearn.cluster import AgglomerativeClustering
from config import FMRI_RDMS


def main(args):
    subject = args.subject
    nsd_dataset = DatasetNSD_872(args.data_dir)
    paths = nsd_dataset.load_dataset(args.data_dir)

    for sensitivity in FMRI_RDMS.keys():
        for area in FMRI_RDMS[sensitivity].keys():
            logger.info(f"Processing {sensitivity} - {area} for subject {subject}")
            rdm_path = os.path.join(
                paths["NSD_872_RDMs"], FMRI_RDMS[sensitivity][area].format(subject)
            )
            rdm_fmri = np.load(rdm_path)["rdm"]

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.9,
                linkage="average",
                metric="precomputed",
            ).fit(rdm_fmri)

            sorted_indices = np.argsort(clustering.labels_)
            rdm_fmri_sorted = rdm_fmri[sorted_indices][:, sorted_indices]

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f"Subject {subject} - {sensitivity} - {area}", fontsize=16)

            vmin = rdm_fmri.min()
            vmax = rdm_fmri.max()

            sns.heatmap(
                rdm_fmri,
                ax=axes[0],
                square=True,
                xticklabels=False,
                yticklabels=False,
                cbar=False,
                vmin=vmin,
                vmax=vmax,
            )
            axes[0].set_title("Before Clustering")

            sns.heatmap(
                rdm_fmri_sorted,
                ax=axes[1],
                square=True,
                xticklabels=False,
                yticklabels=False,
                cbar=True,
                vmin=vmin,
                vmax=vmax,
            )
            axes[1].set_title("After Clustering")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # --- Saving ---
            plots_dir = args.plots_dir
            indices_dir = args.indices_dir

            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(indices_dir, exist_ok=True)

            base_filename = f"subject_{subject}_{sensitivity}_{area}"
            plot_path = os.path.join(plots_dir, f"{base_filename}.png")
            indices_path = os.path.join(indices_dir, f"{base_filename}_indices.npy")

            plt.savefig(plot_path)
            np.save(indices_path, sorted_indices)
            plt.close(fig)

            logger.info(f"Saved plot to {plot_path}")
            logger.info(f"Saved sorted indices to {indices_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster fMRI RDMs")
    parser.add_argument(
        "--data_dir", type=str, default="./", help="Path to the NSD dataset"
    )
    parser.add_argument("--subject", type=int, default=1, help="Subject number")
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="plots/fmri_rdm_clustering",
        help="Directory for saving plots",
    )
    parser.add_argument(
        "--indices_dir",
        type=str,
        default="clustered_indices",
        help="Directory for saving sorted indices",
    )
    args = parser.parse_args()

    main(args)
