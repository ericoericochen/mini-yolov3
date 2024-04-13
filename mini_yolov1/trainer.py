import torch
from .model import YOLO
from .dataset import ObjectDetectionDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .evals import calculate_mAP, calculate_loss
import os
import json
import pprint
import matplotlib.pyplot as plt
from .loss import YOLOLoss
from .utils import draw_bounding_boxes, to_tensor
from torchvision.ops import box_convert
import torch.nn.functional as F


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(
        self,
        model: YOLO,
        train_dataset: ObjectDetectionDataset,
        val_dataset: ObjectDetectionDataset = None,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        num_epochs: int = 10,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        log_detections: bool = True,
        save_dir: str = "./yolo_checkpoints",
        checkpoint_epoch: int = 1,
        eval_every: int = 1,
        device: str = get_device(),
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.save_dir = save_dir
        self.log_detections = log_detections
        self.checkpoint_epoch = checkpoint_epoch
        self.eval_every = eval_every
        self.device = device

    def record_object_detection_results(self, results_dir: str, epoch: int):
        self.model.eval()

        for i in range(4):
            train_save_path = os.path.join(results_dir, f"train_{epoch}_{i}.png")
            train_image = self.train_dataset.get_original_image(i)
            train_image_tensor = (
                self.train_dataset[i]["image"].unsqueeze(0).to(self.device)
            )

            bounding_boxes = self.model.inference(train_image_tensor).bboxes

            train_bbox = draw_bounding_boxes(
                train_image,
                box_convert(
                    bounding_boxes[0]["bboxes"],
                    in_fmt="cxcywh",
                    out_fmt="xyxy",
                ),
                [
                    self.train_dataset.LABELS[label.item()]
                    for label in bounding_boxes[0]["labels"]
                ],
                scores=bounding_boxes[0]["scores"],
            )

            plt.clf()
            plt.imshow(train_bbox)
            plt.savefig(train_save_path)

        if self.val_dataset:
            for i in range(4):
                val_save_path = os.path.join(results_dir, f"val_{epoch}_{i}.png")
                val_image = self.val_dataset.get_original_image(i)
                val_image_tensor = (
                    self.val_dataset[i]["image"].unsqueeze(0).to(self.device)
                )
                bounding_boxes = self.model.inference(val_image_tensor).bboxes

                val_bbox = draw_bounding_boxes(
                    val_image,
                    box_convert(
                        bounding_boxes[0]["bboxes"],
                        in_fmt="cxcywh",
                        out_fmt="xyxy",
                    ),
                    [
                        self.train_dataset.LABELS[label]
                        for label in bounding_boxes[0]["labels"]
                    ],
                    scores=bounding_boxes[0]["scores"],
                )

                plt.clf()
                plt.imshow(val_bbox)
                plt.savefig(val_save_path)

    def train(self):
        # make save dir
        os.makedirs(self.save_dir, exist_ok=True)
        checkpoints_dir = os.path.join(self.save_dir, "checkpoints")
        results_dir = os.path.join(self.save_dir, "results")
        model_config_path = os.path.join(self.save_dir, "model_config.json")
        train_config_path = os.path.join(self.save_dir, "train_config.json")
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # save train config
        with open(train_config_path, "w") as f:
            json.dump(
                {
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.train_loader.batch_size,
                    "num_epochs": self.num_epochs,
                    "lambda_coord": self.lambda_coord,
                    "lambda_noobj": self.lambda_noobj,
                },
                f,
            )

        # save model config
        with open(model_config_path, "w") as f:
            json.dump(self.model.config, f)

        loss_plot_path = os.path.join(self.save_dir, "loss.png")

        # set up evals json to track mAP
        evals_json = {}
        evals_path = os.path.join(self.save_dir, "evals.json")

        with open(evals_path, "w") as f:
            json.dump({}, f)

        # prepare model and optimizer
        num_iters = len(self.train_loader) * self.num_epochs

        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        train_losses = []
        val_losses = []
        train_mAPs = []
        val_mAPs = []

        has_val = self.val_dataset is not None
        criterion = YOLOLoss(
            S=model.S,
            B=model.B,
            C=model.C,
            lambda_coord=self.lambda_coord,
            lambda_noobj=self.lambda_noobj,
        )

        pp = pprint.PrettyPrinter()

        with tqdm(total=num_iters, position=0) as pbar:
            for epoch in range(self.num_epochs):
                epoch_loss = 0

                model.train()
                for data in self.train_loader:
                    # move batch to device
                    images, bboxes, labels = (
                        data["images"],
                        data["bboxes"],
                        data["labels"],
                    )

                    images = images.to(self.device)
                    bboxes = [bbox.to(self.device) for bbox in bboxes]
                    labels = [label.to(self.device) for label in labels]

                    # make pred and compute loss
                    pred = model(images)
                    loss, loss_breakdown = criterion(pred, bboxes, labels)

                    epoch_loss += loss.item()

                    # back prop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.update(1)
                    pbar.set_postfix(
                        loss=loss.item(),
                        **loss_breakdown,
                    )

                    train_losses.append(loss.item())

                epoch_loss /= len(self.train_loader)

                if has_val:
                    val_loss = calculate_loss(
                        model, self.val_loader, criterion, device=self.device
                    )
                    val_losses.append(val_loss)
                    tqdm.write(
                        f"[Epoch {epoch}] Train Loss: {epoch_loss} | Val Loss: {val_loss}"
                    )
                else:
                    tqdm.write(f"[Epoch {epoch}] Train Loss: {epoch_loss}")

                # visualize object detection results on train and val
                if self.log_detections:
                    self.record_object_detection_results(results_dir, epoch)

                # save loss plot
                plt.clf()
                plt.title("Log Loss")
                plt.semilogy(train_losses, label="Train Loss")
                if has_val:
                    plt.semilogy(
                        [(i + 1) * len(self.train_loader) for i in range(epoch + 1)],
                        val_losses,
                        label="Val Loss",
                    )

                plt.legend()
                plt.savefig(loss_plot_path)

                # evaluate mAP
                if (epoch + 1) % self.eval_every == 0:
                    tqdm.write(f"[INFO] Evals Epoch {epoch}")
                    epoch_eval_data = {}

                    print("[INFO] Calculating Train mAP")
                    train_mAP = calculate_mAP(
                        model, self.train_loader, device=self.device
                    )
                    epoch_eval_data["train_mAP"] = train_mAP
                    train_mAPs.append(train_mAP["map_50"])

                    tqdm.write(f"Train mAP: {pp.pformat(train_mAP)}")

                    if has_val:
                        print("[INFO] Calculating Val mAP")
                        val_mAP = calculate_mAP(
                            model, self.val_loader, device=self.device
                        )
                        epoch_eval_data["val_mAP"] = val_mAP
                        val_mAPs.append(val_mAP["map_50"])

                        tqdm.write(f"Val mAP: {pp.pformat(val_mAP)}")

                    evals_json[epoch] = epoch_eval_data
                    with open(evals_path, "w") as f:
                        json.dump(evals_json, f)

                    plt.clf()
                    plt.title("mAP")
                    plt.plot(
                        [i * self.eval_every for i in range(len(train_mAPs))],
                        train_mAPs,
                        label="Train mAP",
                    )

                    if has_val:
                        plt.plot(
                            [i * self.eval_every for i in range(len(val_mAPs))],
                            val_mAPs,
                            label="Val mAP",
                        )
                    plt.savefig(os.path.join(self.save_dir, "mAP.png"))

                if (epoch + 1) % self.checkpoint_epoch == 0:
                    weights_path = os.path.join(checkpoints_dir, f"weights_{epoch}.pt")
                    torch.save(model.state_dict(), weights_path)

        weights_path = os.path.join(self.save_dir, f"weights.pt")
        torch.save(model.state_dict(), weights_path)

        return train_losses
