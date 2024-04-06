import argparse
import sys

sys.path.append("../")

from mini_yolov3.dataset import SVHNDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="../configs/smol.json")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str)

    return parser.parse_args()


def main(args):
    train_dataset = SVHNDataset(split="train")
    val_dataset = SVHNDataset(split="val")


if __name__ == "__main__":
    args = parse_args()
    main(args)
