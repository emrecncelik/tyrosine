# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 18:28:17 2025

@author: sarat
"""

#%% imports and functions
import numpy as np
import matplotlib.pyplot as plt
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

#%% similarity between fMRI RDM and image and text RDM for different subjects in different brain areas
rdm_text = np.load('text_RDM.npy')
rdm_images = np.load('image_RDM.npy')
df_rdm_similarities = pd.DataFrame()
fmri_rdm_folder = "C:/Users/sarat/Net2Brain/notebooks/NSD Dataset/NSD_872_RDMs"
for region_name in os.listdir(fmri_rdm_folder):
    region_folder = os.path.join(fmri_rdm_folder, region_name)
    print(region_folder)
    if not os.path.isdir(region_folder): continue
    for area_filename in os.listdir(os.path.join(region_folder, "combined")):
        if not area_filename.endswith(".npz"): 
            continue
        area_name = area_filename.split("_")[0]
        print(area_name)
        rdm_fmri_combined = np.load(
            os.path.join(region_folder, "combined", area_filename))["rdm"]
        for s, rdm_fmri_subject in enumerate(rdm_fmri_combined):
            sim_images_subject = compute_RDM_similarity(
                rdm_images, rdm_fmri_subject)
            sim_text_subject = compute_RDM_similarity(
                rdm_text, rdm_fmri_subject)
            similarities_image = {'region': [region_name], 
                            'area': [area_name], 
                            'subject_id': [s+1], 
                            'comparison': 'image',
                            'sim': [sim_images_subject]}
            similarities_text = {'region': [region_name], 
                            'area': [area_name], 
                            'subject_id': [s+1], 
                            'comparison': 'text','sim': [sim_text_subject]}
            df_rdm_similarities = pd.concat((
                df_rdm_similarities, pd.DataFrame(similarities_image)), 
                axis=0) 
            df_rdm_similarities = pd.concat((
                df_rdm_similarities, pd.DataFrame(similarities_text)), 
                axis=0) 
            
# plot rdm similarities by area
num_areas = len(np.unique(df_rdm_similarities['area']))
fig, ax = plt.subplots(1, 1, figsize=(num_areas, 3))
sns.stripplot(data=df_rdm_similarities, x='area', y='sim', hue='comparison', dodge=True)
sns.pointplot(data=df_rdm_similarities, x='area', y='sim', hue='comparison', dodge=0.25, join=False)
sns.despine()
plt.tight_layout()
