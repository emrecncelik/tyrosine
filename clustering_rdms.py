# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 17:12:00 2025

@author: sarat
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import os
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns

def cluster_RDM(rdm, clustered_indices):
    clustered_rdm = rdm[:, clustered_indices]
    clustered_rdm = clustered_rdm[clustered_indices, :]
    return clustered_rdm

def compute_RDM_similarity(rdm_A, rdm_B, metric="pearsonr"):
    if metric == "cosine":
        rdm_sim = np.dot(rdm_A.ravel(), rdm_B.ravel()) /\
            (np.linalg.norm(rdm_A.ravel()) * np.linalg.norm(rdm_B.ravel()))
    elif metric == "pearsonr":
        rdm_sim = pearsonr(rdm_A.ravel(), rdm_B.ravel()).statistic
    return float(rdm_sim)

#%% get image and text features from pretrained CLIP model
# read image names and labels
images_folder = "C:/Users/sarat/Net2Brain/notebooks/NSD Dataset/NSD_872_images"
images = os.listdir(images_folder)[1:-1]

captions_folder = "C:/Users/sarat/Net2Brain/notebooks/NSD Dataset/NSD_872_captions"
captions = {}
for file in os.listdir(captions_folder):
    if '.txt' not in file: continue
    with open(os.path.join(captions_folder, file)) as f:
        caption = f.read()
    captions[file[:-4]] = caption
    
# get image and text features from clip model
from transformers import AutoTokenizer, AutoProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

language_features = []
for file, caption in captions.items():
    text_inputs = tokenizer([caption], padding=True, return_tensors="pt")
    text_features = model.get_text_features(**text_inputs).detach().cpu().numpy()
    language_features.append(text_features)
language_features = np.array(language_features).squeeze()

vision_features = []
for file in images:
    image = Image.open(os.path.join(images_folder, file))
    image_inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**image_inputs).detach().cpu().numpy()
    vision_features.append(image_features)
vision_features = np.array(vision_features).squeeze()

# RDM of text and images
rdm_text = 1-np.corrcoef(language_features)
rdm_images = 1-np.corrcoef(vision_features)
np.save('image_RDM.npy', rdm_text)
np.save('text_RDM.npy', rdm_images)

# image_rdm_clustering = AgglomerativeClustering(
#     n_clusters=None, distance_threshold=0.5, linkage="average", metric="precomputed").fit(rdm_images)
# print(image_rdm_clustering.labels_.max())
# shuffled_indices_image = []
# for label in range(image_rdm_clustering.labels_.max()+1):
#     label_indices = np.where(image_rdm_clustering.labels_==label)[0]
#     shuffled_indices_image.extend(label_indices)
# shuffled_indices_image = np.array(shuffled_indices_image, int)

#%% RDM comparisons

# fMRI data
# fmri_data_ffa1_lh = np.load(
#     "C:/Users/sarat/Net2Brain/notebooks/NSD Dataset/NSD_872_fmri/floc-faces/FFA-1/subj01_roi-FFA-1_lh.npy")
# fmri_data_ffa1_rh = np.load(
#     "C:/Users/sarat/Net2Brain/notebooks/NSD Dataset/NSD_872_fmri/floc-faces/FFA-1/subj01_roi-FFA-1_rh.npy")
# fmri_data_eba_both = np.concatenate((fmri_data_ffa1_lh, fmri_data_ffa1_rh), axis=1)
# rdm_fmri = 1-np.corrcoef(fmri_data_eba_both)
fmri_rdm_filename = "C:/Users/sarat/Net2Brain/notebooks/NSD Dataset/NSD_872_RDMs/floc-bodies/subject_1_merged_EBA_both.npz"
rdm_fmri = np.load(
    fmri_rdm_filename)["rdm"]
plt.figure()
plt.imshow(rdm_fmri)

# clustering fMRI RDMs
fmri_rdm_clustering = AgglomerativeClustering(
    n_clusters=None, distance_threshold=0.9, linkage="average", 
    metric="precomputed").fit(rdm_fmri)
print(fmri_rdm_clustering.labels_.max())
# shuffle fMRI RDM to cluster them based on labels
shuffled_indices_fmri = []
for label in range(fmri_rdm_clustering.labels_.max()+1):
    label_indices = np.where(fmri_rdm_clustering.labels_==label)[0]
    shuffled_indices_fmri.extend(label_indices)
clustered_indices_fmri = np.array(shuffled_indices_fmri, int)
rdm_fmri_clustered = cluster_RDM(rdm_fmri, clustered_indices_fmri)

# use fMRI RDM cluster labels to shuffle image and text RDMs
rdm_images_clustered = cluster_RDM(rdm_images, clustered_indices_fmri)
sim_images = compute_RDM_similarity(rdm_images_clustered, rdm_fmri_clustered)
rdm_text_clustered = cluster_RDM(rdm_text, clustered_indices_fmri)
sim_text = compute_RDM_similarity(rdm_fmri_clustered, rdm_text_clustered)

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[0].imshow(rdm_fmri_clustered)
ax[0].set_title(f"fMRI RDM {fmri_rdm_filename.split('/')[-1][:-4]}")
ax[1].imshow(rdm_images_clustered)
ax[1].set_title(f'image RDM, {sim_images:.2f}')
ax[2].imshow(rdm_text_clustered)
ax[2].set_title(f'text RDM, {sim_text:.2f}')
