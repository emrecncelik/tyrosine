import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rsatoolbox.rdm.compare import compare
from rsatoolbox.rdm import RDMs
from utils import to_distance_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sims = pd.read_csv("similarities.csv")

sims_long = sims.melt(
    id_vars=["subject", "sensitivity", "area"],
    value_vars=["CLIP_vis", "CLIP_lang", "ViT", "BERT"],
    var_name="model",
    value_name="similarity",
)


def model2model(txt):
    if "CLIP_vis" in txt:
        return "CLIP Vision"
    if "CLIP_lang" in txt:
        return "CLIP Language"
    else:
        return txt


def model2modality(txt):
    if "CLIP Vision" in txt:
        return "Vision"
    elif "CLIP Language" in txt:
        return "Language"
    elif "ViT" in txt:
        return "Vision"
    elif "BERT" in txt:
        return "Language"
    else:
        return txt


sims_long["model"] = sims_long["model"].apply(model2model)
sims_long["modality"] = sims_long["model"].apply(model2modality)
# sims_long.columns = [c.capitalize() for c in sims_long.columns]
sims_long["sensitivity"] = sims_long["sensitivity"].str.capitalize()


sns.reset_defaults()
plt.rcdefaults()

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 14,
    }
)

g = sns.FacetGrid(
    sims_long,
    col="sensitivity",
    col_wrap=3,
    height=6,
    aspect=0.9,
    sharex=False,
    sharey=True,
)

# g.map_dataframe(
#     sns.stripplot,
#     x="area",
#     y="similarity",
#     hue="model",
#     dodge=True,
#     jitter=True,
#     size=10,
#     alpha=0.8,
#     palette="deep",
# )
g.map_dataframe(
    sns.pointplot,
    x="area",
    y="similarity",
    hue="model",
    dodge=0.6,
    errorbar="se",
    markers="d",
    capsize=0.1,
    # linestyle='none',
    palette="deep",
    # color='black',
    markersize=8,
)
g.add_legend(
    title="Model",
    bbox_to_anchor=(0.9, 1.1),
    loc="upper left",
    title_fontsize=14,
    fontsize=12,
)

g.set_axis_labels("Region of Interest (ROI)", "Correlation", fontsize=14)
g.set_titles("{col_name}", fontsize=16)

for ax in g.axes.flat:
    ax.tick_params(axis="x", rotation=30, labelsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7, color="gray")

for ax in g.axes.flat:
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(16)
        item.set_fontfamily("serif")

for ax in g.axes.flat:
    ax.tick_params(axis="x", rotation=30, labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7, color="gray")
    ax.set_yticks([i * 0.05 for i in range(6)])  #

for ax in g.axes.flat:
    legend = ax.get_legend()
    if legend:
        legend.get_title().set_fontsize(18)
        legend.get_title().set_fontfamily("serif")
        for text in legend.get_texts():
            text.set_fontsize(18)
            text.set_fontfamily("serif")

plt.suptitle(
    "Similarity of Model Representations to Brain Regions", y=1.02, fontsize=18
)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(
    "plots/similarity_plot.png", dpi=300, bbox_inches="tight", transparent=False
)
plt.show()
