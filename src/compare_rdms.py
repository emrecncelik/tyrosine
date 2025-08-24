import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rsatoolbox.rdm.compare import compare
from rsatoolbox.rdm import RDMs
from utils import get_fmri_rdm
from config import FMRI_RDMS

lang_rdm = np.load(
    "rdms_modality/ViT-L_-_14/language/RDM_transformer_resblocks_11.npz",
    allow_pickle=True,
)
vis_rdm = np.load(
    "rdms_modality/ViT-L_-_14/vision/RDM_visual_transformer_resblocks_23.npz",
    allow_pickle=True,
)
bert_rdm = np.load(
    "rdms_modality/bert-base-uncased/language/RDM_encoder_layer_11.npz",
    allow_pickle=True,
)
vit_rdm = np.load(
    "rdms_modality/vit_large_patch16_224_in21k/vision/RDM_blocks_23.npz",
    allow_pickle=True,
)

lang_rdm = RDMs(
    lang_rdm["rdm"], rdm_descriptors={"model": "CLIP", "modality": "language"}
)
vis_rdm = RDMs(vis_rdm["rdm"], rdm_descriptors={"model": "CLIP", "modality": "vision"})
bert_rdm = RDMs(
    bert_rdm["rdm"], rdm_descriptors={"model": "BERT", "modality": "language"}
)
vit_rdm = RDMs(vit_rdm["rdm"], rdm_descriptors={"model": "ViT", "modality": "vision"})

similarities = []

for subject in range(1, 9):
    for sensitivity in FMRI_RDMS.keys():
        for area in FMRI_RDMS[sensitivity].keys():
            try:
                rdm = get_fmri_rdm(
                    subject=subject,
                    sensitivity=sensitivity,
                    area=area,
                    nsd_dir="/media/stuff/neuroai_data",
                ).reshape(1, 872, 872)
                rdm = RDMs(
                    rdm,
                    rdm_descriptors={
                        "subject": subject,
                        "sensitivity": sensitivity,
                        "area": area,
                    },
                )
            except FileNotFoundError as e:
                print(
                    f"RDM doesn't exist for subject {subject}, sensitivity {sensitivity}, area {area}: {e}"
                )
                continue

            similarity_vis = compare(rdm, vis_rdm, method="corr")
            similarity_lang = compare(rdm, lang_rdm, method="corr")
            similarity_vit = compare(rdm, vit_rdm, method="corr")
            similarity_bert = compare(rdm, bert_rdm, method="corr")

            similarity = {
                "subject": subject,
                "sensitivity": sensitivity,
                "area": area,
                "CLIP_vis": similarity_vis.item(),
                "CLIP_lang": similarity_lang.item(),
                "ViT": similarity_vit.item(),
                "BERT": similarity_bert.item(),
            }

            similarities.append(similarity)

similarities = pd.DataFrame(similarities)
similarities.to_csv("similarities.csv", index=False)
