import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import get_fmri_rdm, get_fmri_indices, to_distance_matrix
from config import FMRI_RDMS
from loguru import logger


def main(args):
    os.makedirs(args.plots_dir, exist_ok=True)

    for sensitivity in FMRI_RDMS.keys():
        for area in FMRI_RDMS[sensitivity].keys():
            logger.info(f"Processing {sensitivity} - {area} for subject {args.subject}")

            # Load RDMs
            lang_rdm = np.load(
                os.path.join(
                    args.rdms_dir,
                    "ViT-L_-_14/language/RDM_transformer_resblocks_11.npz",
                ),
                allow_pickle=True,
            )
            vis_rdm = np.load(
                os.path.join(
                    args.rdms_dir,
                    "ViT-L_-_14/vision/RDM_visual_transformer_resblocks_23.npz",
                ),
                allow_pickle=True,
            )
            bert_rdm = np.load(
                os.path.join(
                    args.rdms_dir,
                    "bert-base-uncased/language/RDM_encoder_layer_11.npz",
                ),
                allow_pickle=True,
            )
            vit_rdm = np.load(
                os.path.join(
                    args.rdms_dir,
                    "vit_large_patch16_224_in21k/vision/RDM_blocks_23.npz",
                ),
                allow_pickle=True,
            )

            lang_rdm_matrix = to_distance_matrix(
                torch.from_numpy(lang_rdm["rdm"])
            ).numpy()
            vis_rdm_matrix = to_distance_matrix(
                torch.from_numpy(vis_rdm["rdm"])
            ).numpy()
            bert_rdm_matrix = to_distance_matrix(
                torch.from_numpy(bert_rdm["rdm"])
            ).numpy()
            vit_rdm_matrix = to_distance_matrix(
                torch.from_numpy(vit_rdm["rdm"])
            ).numpy()

            fmri_rdm = get_fmri_rdm(
                args.subject,
                sensitivity=sensitivity,
                area=area,
                nsd_dir=args.nsd_dir,
            )
            indices = get_fmri_indices(
                args.subject,
                sensitivity=sensitivity,
                area=area,
                indices_dir=args.indices_dir,
            )

            fig, ax = plt.subplots(1, 5, figsize=(24, 12))

            ax[0].set_title(f"Brain area: {area}, Subject: {args.subject}")
            ax[0].imshow(fmri_rdm[indices][:, indices], cmap=args.cmap)
            ax[1].set_title("CLIP language")
            ax[1].imshow(lang_rdm_matrix[indices][:, indices], cmap=args.cmap)
            ax[2].set_title("CLIP vision")
            ax[2].imshow(vis_rdm_matrix[indices][:, indices], cmap=args.cmap)
            ax[3].set_title("BERT (language-only)")
            ax[3].imshow(bert_rdm_matrix[indices][:, indices], cmap=args.cmap)
            ax[4].set_title("ViT (vision-only)")
            ax[4].imshow(vit_rdm_matrix[indices][:, indices], cmap=args.cmap)
            plt.tight_layout()

            save_path = os.path.join(
                args.plots_dir,
                f"subject_{args.subject}_{sensitivity}_{area}_model_comparison.png",
            )
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved plot to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visually compare representational dissimilarity matrices clustered by fMRI RDMs"
    )
    parser.add_argument("--subject", type=int, default=1, help="Subject number")
    parser.add_argument(
        "--cmap", type=str, default="inferno", help="Colormap for the plots"
    )
    parser.add_argument(
        "--nsd_dir", type=str, default="./", help="Path to the NSD dataset"
    )
    parser.add_argument(
        "--indices_dir",
        type=str,
        default="clustered_indices",
        help="Directory for loading sorted indices",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="plots/rdm_comparison",
        help="Directory for saving plots",
    )
    parser.add_argument(
        "--rdms_dir",
        type=str,
        default="rdms_modality",
        help="Directory containing the model RDMs",
    )

    args = parser.parse_args()
    main(args)
