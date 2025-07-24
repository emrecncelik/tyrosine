# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:30:31 2025

@author: sarat
"""

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os 
from sklearn.cluster import AgglomerativeClustering

#%% 
df_categories = pd.read_csv('cats.csv')
cat_supercat_labels = [f'{cat}/{supercat}' for cat, supercat in zip(df_categories['name'].values, df_categories['supercategory'].values)]
df_stimuli_ordering = pd.read_csv('stimuli_ordered.csv')
df_stimuli_ordering = df_stimuli_ordering.sort_values(by=['nsd_id'])
category_ids = df_stimuli_ordering['category_id'].values
integer_category_ids = []
for cat_ids in category_ids:
    integer_category_ids.append([int(cat_id) for cat_id in cat_ids.split('|')])
    
fmri_rdm_folder = "C:/Users/sarat/Net2Brain/notebooks/NSD Dataset/NSD_872_RDMs"
subject = "subject_8"
for region_name in os.listdir(fmri_rdm_folder):
    region_folder = os.path.join(fmri_rdm_folder, region_name)
    print(region_folder)
    if not os.path.isdir(region_folder): continue
    fig, ax = plt.subplots(len(os.listdir(os.path.join(region_folder, "combined")))-1, 1, 
                           figsize=(10, 8), sharex=True, num=f'{subject}_{region_name}')    
    a=-1
    for area_filename in (os.listdir(os.path.join(region_folder))):
        if not area_filename.startswith(subject): 
            continue
        a=a+1
        area_name = area_filename.split("_")[3]
        print(area_name)
        rdm_fmri_subject = np.load(
            os.path.join(region_folder, area_filename))["rdm"]
        #for s, rdm_fmri_subject in enumerate(rdm_fmri_combined):
        # clustering
        fmri_rdm_clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.95, linkage="average", 
            metric="precomputed").fit(rdm_fmri_subject)
        print(fmri_rdm_clustering.labels_.max())
        cluster_labels = fmri_rdm_clustering.labels_#np.random.randint(0, 22, (872,))
        df_label_category_ids = pd.DataFrame()
        for label in np.unique(cluster_labels):
            image_idx_label = np.where(cluster_labels==label)[0]
            for image_idx in image_idx_label:
                image_label_category_ids = integer_category_ids[image_idx]
                df_label_category_ids = pd.concat(
                    (df_label_category_ids, 
                     pd.DataFrame(
                         {'label': [label]*len(image_label_category_ids), 
                          'image_id': [image_idx]*len(image_label_category_ids), 
                          'category_id': image_label_category_ids})))
        bins = np.arange(df_label_category_ids['category_id'].min(), 
                       df_label_category_ids['category_id'].max())
        sns.histplot(data=df_label_category_ids, x='category_id', hue='label', ax=ax[a], palette='tab20', 
                    bins=bins,  
                    )
        ax[a].set_title(f'{area_name}, num_clusters={fmri_rdm_clustering.labels_.max()}')
        ax[a].legend([])
        sns.despine()
    ax[-1].set_xticks(df_categories['id'].values)
    ax[-1].set_xticklabels(df_categories['supercategory'], rotation=90)
    plt.suptitle(region_name)
    plt.tight_layout()   


#%%
cluster_labels = fmri_rdm_clustering.labels_#np.random.randint(0, 22, (872,))
df_label_category_ids = pd.DataFrame()
for label in np.unique(cluster_labels):
    image_idx_label = np.where(cluster_labels==label)[0]
    for image_idx in image_idx_label:
        image_label_category_ids = integer_category_ids[image_idx]
        df_label_category_ids = pd.concat(
            (df_label_category_ids, 
             pd.DataFrame(
                 {'label': [label]*len(image_label_category_ids), 
                  'image_id': [image_idx]*len(image_label_category_ids), 
                  'category_id': image_label_category_ids})))
        
fig, ax = plt.subplots()
bins = np.arange(df_label_category_ids['category_id'].min(), 
               df_label_category_ids['category_id'].max())
sns.histplot(data=df_label_category_ids, x='category_id', hue='label', ax=ax, palette='tab20', 
            bins=bins,  
            )
ax.set_xticks(df_categories['id'].values)
ax.set_xticklabels(df_categories['supercategory'].values, rotation=90)
plt.tight_layout()