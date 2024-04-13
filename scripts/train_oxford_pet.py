import argparse
import sys
import json
import torch
import os

sys.path.append("../")

from mini_yolov1.dataset import OxfordIIITPetDataset, RandomColorJitter
from mini_yolov1.model import YOLO
from mini_yolov1.trainer import Trainer
from torchvision.transforms import v2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/oxford-iiit-pet")
    parser.add_argument("--model_config", type=str, default="../configs/smol.json")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lambda_coord", type=float, default=5.0)
    parser.add_argument("--lambda_noobj", type=float, default=0.5)
    parser.add_argument("--data_augment", action="store_true", default=False)
    parser.add_argument("--augment_prob", type=float, default=0.25)
    parser.add_argument("--checkpoint_epoch", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_dir", type=str, required=True)

    return parser.parse_args()


def main(args):
    torch.manual_seed(0)

    print("[INFO] Training Mini Yolo V3 on SVHN...")
    print(args)

    dataset_transform = v2.Compose(
        [
            v2.Resize((args.image_size, args.image_size)),
            RandomColorJitter(prob=args.augment_prob),
            v2.ToTensor(),
        ]
    )
    train_dataset = OxfordIIITPetDataset(
        root=args.data_dir,
        split="trainval",
        transform=dataset_transform,
    )

    print(f"[INFO] Train Dataset Size: {len(train_dataset)}")

    # save args
    os.makedirs(args.save_dir, exist_ok=True)
    with open(f"{args.save_dir}/args.json", "w") as f:
        json.dump(vars(args), f)

    val_dataset = OxfordIIITPetDataset(
        root=args.data_dir, split="test", transform=dataset_transform
    )

    print(f"[INFO] Val Dataset Size: {len(val_dataset)}")

    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    model = YOLO(**model_config)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        lambda_coord=args.lambda_coord,
        lambda_noobj=args.lambda_noobj,
        save_dir=args.save_dir,
        checkpoint_epoch=args.checkpoint_epoch,
        eval_every=args.eval_every,
        device="cpu",
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
