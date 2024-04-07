import torch
from .model import MiniYoloV3
from .dataset import ObjectDetectionDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .evals import calculate_mAP, calculate_loss
import os
import json
import pprint


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


class Trainer:
    def __init__(
        self,
        model: MiniYoloV3,
        train_dataset: ObjectDetectionDataset,
        val_dataset: ObjectDetectionDataset = None,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        num_epochs: int = 10,
        lambda_coord: float = 1.0,
        lambda_noobj: float = 1.0,
        save_dir: str = "./yolo_checkpoints",
        checkpoint_epoch: int = 1,
        eval_every: int = 1,
        device: str = get_device(),
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.save_dir = save_dir
        self.checkpoint_epoch = checkpoint_epoch
        self.eval_every = eval_every
        self.device = device

    def train(self):
        # make save dir
        os.makedirs(self.save_dir, exist_ok=True)
        checkpoints_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # set up evals json to track mAP
        evals_json = {}
        evals_path = os.path.join(self.save_dir, "evals.json")

        with open(evals_path, "w") as f:
            json.dump({}, f)

        # prepare model and optimizer
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        losses = []
        val_losses = []
        model.train()
        pp = pprint.PrettyPrinter()
        num_iters = len(self.train_loader) * self.num_epochs

        with tqdm(total=num_iters, position=0) as pbar:
            for epoch in range(self.num_epochs):
                epoch_loss = 0

                for data in self.train_loader:
                    images, bboxes, labels = (
                        data["images"],
                        data["bboxes"],
                        data["labels"],
                    )

                    images = images.to(self.device)
                    bboxes = [bbox.to(self.device) for bbox in bboxes]
                    labels = [label.to(self.device) for label in labels]

                    # make pred and compute loss
                    loss, loss_breakdown = model.get_yolo_loss(
                        images=images,
                        bboxes=bboxes,
                        labels=labels,
                        lambda_coord=self.lambda_coord,
                        lambda_noobj=self.lambda_noobj,
                    )

                    epoch_loss += loss.item()

                    # back prop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.update(1)
                    pbar.set_postfix(
                        loss=loss.item(),
                        **{k: v.item() for k, v in loss_breakdown.items()},
                    )

                epoch_loss /= len(self.train_loader)
                losses.append(epoch_loss)

                if self.val_loader:
                    val_loss = calculate_loss(
                        model, self.val_loader, device=self.device
                    )

                    val_losses.append(val_loss)

                    tqdm.write(
                        f"[Epoch {epoch}] Train Loss: {epoch_loss} | Val Loss: {val_loss}"
                    )
                else:
                    tqdm.write(f"[Epoch {epoch}] Loss: {epoch_loss}")

                if (epoch + 1) % self.eval_every == 0:
                    tqdm.write(f"[INFO] Evals Epoch {epoch}")
                    epoch_eval_data = {}

                    print("[INFO] Calculating Train mAP")
                    train_mAP = calculate_mAP(
                        model, self.train_loader, device=self.device
                    )
                    epoch_eval_data["train_mAP"] = train_mAP

                    tqdm.write(f"Train mAP: {pp.pformat(train_mAP)}")

                    if self.val_loader:
                        print("[INFO] Calculating Val mAP")
                        val_mAP = calculate_mAP(
                            model, self.val_loader, device=self.device
                        )
                        epoch_eval_data["val_mAP"] = val_mAP

                        tqdm.write(f"Val mAP: {pp.pformat(val_mAP)}")

                    evals_json[epoch] = epoch_eval_data
                    with open(evals_path, "w") as f:
                        json.dump(evals_json, f)

                if (epoch + 1) % self.checkpoint_epoch == 0:
                    weights_path = os.path.join(checkpoints_dir, f"weights_{epoch}.pt")
                    torch.save(model.state_dict(), weights_path)

        weights_path = os.path.join(self.save_dir, f"weights.pt")
        torch.save(model.state_dict(), weights_path)

        return losses
