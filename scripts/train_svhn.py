import argparse
import sys
import json

sys.path.append("../")

from mini_yolov3.dataset import SVHNDataset
from mini_yolov3.model import MiniYoloV3
from mini_yolov3.trainer import Trainer


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
    train_dataset = SVHNDataset(split="train", image_size=args.image_size)
    val_dataset = SVHNDataset(split="test", image_size=args.image_size)

    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    model = MiniYoloV3.from_config(model_config)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        device="cpu",
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
