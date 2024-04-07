import argparse
import sys
import json
import torch

sys.path.append("../")

from mini_yolov3.dataset import SVHNDataset
from mini_yolov3.model import MiniYoloV3
from mini_yolov3.trainer import Trainer
from torch.utils.data import Subset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="../configs/smol.json")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lambda_coord", type=float, default=0.05)
    parser.add_argument("--lambda_conf", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=0.5)
    parser.add_argument("--checkpoint_epoch", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_dir", type=str, required=True)

    return parser.parse_args()


def main(args):
    torch.manual_seed(0)

    print("[INFO] Training Mini Yolo V3 on SVHN...")

    train_dataset = SVHNDataset(split="train", image_size=args.image_size)
    train_dataset = Subset(train_dataset, range(1000))

    val_dataset = SVHNDataset(split="test", image_size=args.image_size)

    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    model = MiniYoloV3(**model_config)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        lambda_coord=args.lambda_coord,
        lambda_conf=args.lambda_conf,
        lambda_cls=args.lambda_cls,
        save_dir=args.save_dir,
        checkpoint_epoch=args.checkpoint_epoch,
        eval_every=args.eval_every,
        device="cpu",
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
