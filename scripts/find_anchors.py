import argparse
import sys
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np


sys.path.append("../")

from mini_yolov3.dataset import SVHNDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_anchors", type=int, default=6)

    return parser.parse_args()


def main(args):
    dataset = SVHNDataset(split="train", image_size=32)

    box_dims = []
    for data in tqdm(dataset):
        bbox = data["bbox"]

        wh = bbox[:, 2:].tolist()
        box_dims += wh

    box_dims = np.array(box_dims)
    kmeans = KMeans(n_clusters=args.num_anchors)
    kmeans.fit(box_dims)

    # get sorted anchor boxes by area
    anchor_boxes = kmeans.cluster_centers_
    areas = np.prod(anchor_boxes, axis=1)
    sorted_indices = np.argsort(-areas)  # Negative for descending order
    sorted_anchor_boxes = anchor_boxes[sorted_indices]

    print(sorted_anchor_boxes)


if __name__ == "__main__":
    args = parse_args()
    main(args)
