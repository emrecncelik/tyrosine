import os
import glob
import json
import argparse
import pandas as pd
from net2brain.utils.download_datasets import DatasetAlgonauts_NSD
from utils import filepath_to_nsd_id


def main(args):
    DATA_DIR = args.data_dir
    NSD_DIR = os.path.join(DATA_DIR, "Algonauts_NSD")

    print(f"Loading NSD dataset from {NSD_DIR}.")
    nsd_dataset = DatasetAlgonauts_NSD(DATA_DIR)
    paths = nsd_dataset.load_dataset(DATA_DIR)

    print("Processing COCO captions and instances.")
    with open(os.path.join(NSD_DIR, "instances_train2017.json"), "r") as f:
        instances = json.load(f)

    with open(os.path.join(NSD_DIR, "captions_train2017.json"), "r") as f:
        captions = json.load(f)

    # DF because easier to manipulate
    captions = pd.DataFrame(captions["annotations"])
    labels = pd.DataFrame(instances["annotations"])

    # Get the common image IDs (1000) for all subjects
    nsd_ids_all_subjects = []
    for i in range(1, 9):
        txt_paths = glob.glob(
            paths[f"subj0{i}"] + "/training_split/training_text/*.txt"
        )
        nsd_ids_all_subjects.append([filepath_to_nsd_id(p) for p in txt_paths])

    nsd_ids_all_subjects = [set(ids) for ids in nsd_ids_all_subjects]
    nsd_ids_intersection = set.intersection(*nsd_ids_all_subjects)
    print(
        f"Number of common NSD stimuli across all subjects: {len(nsd_ids_intersection)}."
    )
    print(
        f"This is different than the original 1000 because Algonauts holds out a subset for testing."
    )

    coco_meta = pd.read_csv(os.path.join(DATA_DIR, "Algonauts_NSD", "coco.csv"))
    coco2nsd = dict(
        zip(coco_meta["cocoId"].astype(int), coco_meta["nsdId"].astype(int))
    )
    common_coco_ids = (
        coco_meta[coco_meta["nsdId"].isin(nsd_ids_intersection)]["cocoId"]
        .astype(int)
        .tolist()
    )

    # Group categories and captions by image_id, make them into a list
    # Multiple categories and captions for a single image exist
    labels1000 = labels[labels["image_id"].isin(common_coco_ids)]
    labels1000 = labels1000.groupby("image_id").agg(lambda x: list(x)).reset_index()
    labels1000 = labels1000[["image_id", "category_id"]]
    labels1000["category_id"] = labels1000["category_id"].apply(
        lambda x: list(sorted(list(set(x))))
    )
    labels1000["category_id"] = labels1000["category_id"].apply(
        lambda x: "|".join(map(str, x))
    )

    captions1000 = captions[captions["image_id"].isin(common_coco_ids)]
    captions1000 = captions1000.groupby("image_id").agg(lambda x: list(x)).reset_index()
    captions1000 = captions1000[["image_id", "caption"]]
    captions1000["caption"] = captions1000["caption"].apply(lambda x: "|".join(x))
    stimuli1000 = pd.merge(captions1000, labels1000, on="image_id")
    stimuli1000["nsd_id"] = stimuli1000["image_id"].apply(
        lambda x: int(coco2nsd[int(x)])
    )
    stimuli1000 = stimuli1000[["image_id", "nsd_id", "caption", "category_id"]]
    stimuli1000.columns = ["coco_id", "nsd_id", "caption", "category_id"]

    stimuli_orders_by_nsd_id_all_subjects = []
    for i in range(1, 9):
        txt_paths = glob.glob(paths[f"subj0{i}_images"] + "/*.png")
        ids = [filepath_to_nsd_id(p) for p in txt_paths]
        order = [
            int(p.split("/")[-1].replace("train-", "").split("_")[0]) for p in txt_paths
        ]

        ids = [x for _, x in sorted(zip(order, ids))]
        stimuli_orders_by_nsd_id_all_subjects.append(
            pd.DataFrame(zip(ids, order), columns=["nsd_id", f"subj0{i}_order"])
        )

    stimuli1000.to_csv(
        os.path.join(args.output_dir, "stimuli872_ordered.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NSD and COCO datasets.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory. "
        "If you have the NSD dataset, this should "
        "be the parent directory containing 'NSD Dataset'.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Path to save the output files."
    )
    args = parser.parse_args()
    main(args)
