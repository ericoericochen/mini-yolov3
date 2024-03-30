import torch
from PIL import Image, ImageDraw
import torchvision
from typing import Union
import matplotlib.pyplot as plt


def coco_to_xyxy_format(bbox: torch.Tensor) -> torch.Tensor:
    """
    Converts COCO format (x_min, y_min, w, h) to (x_min, y_min, x_max, y_max) format

    Params:
        - bbox: (B, 4) in COCO format (x_min, y_min, width, height)
    """

    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    w = bbox[:, 2]
    h = bbox[:, 3]

    x2 = x1 + w
    y2 = y1 + h

    bbox = torch.stack([x1, y1, x2, y2], dim=1)

    return bbox


def coco_to_xywh(bbox: torch.Tensor):
    """
    Converts COCO format (x_min, y_min, w, h) to (x, y, w, h) format

    Params:
        - bbox: (B, 4) in COCO format (x_min, y_min, width, height)
    """

    x = bbox[:, 0] + bbox[:, 2] / 2
    y = bbox[:, 1] + bbox[:, 3] / 2

    bbox = torch.stack([x, y, bbox[:, 2], bbox[:, 3]], dim=1)

    return bbox


def draw_bounding_boxes(
    image: Image.Image,
    bbox: torch.Tensor,
    labels: Union[torch.Tensor, list[str]],
    format="coco",
):
    """
    - bbox: normalized bbox
    """
    if isinstance(labels, torch.Tensor):
        assert len(labels.shape) == 1
        labels = [str(label) for label in labels.tolist()]

    # convert image to be in range [0, 255]
    image = (image * 255).to(torch.uint8)

    bbox = bbox.clone()
    if format == "coco":
        bbox = coco_to_xyxy_format(bbox)

    # rescale bbox to image size
    w, h = image.shape[2], image.shape[1]

    bbox[:, [0, 2]] *= w
    bbox[:, [1, 3]] *= h

    bounding_box_image = torchvision.utils.draw_bounding_boxes(
        image, boxes=bbox, labels=labels
    )
    pil_image = torchvision.transforms.ToPILImage()(bounding_box_image)

    return pil_image


def to_tensor(image: Image.Image):
    return torchvision.transforms.ToTensor()(image)


def to_pil_image(image: torch.Tensor):
    return torchvision.transforms.ToPILImage()(image)


def draw_grid(image, grid_size):
    image = image.copy()

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define the grid parameters
    grid_color = (0, 0, 0)  # Color of the grid lines (RGB)

    # Get the image size
    width, height = image.size

    # Draw the horizontal grid lines
    thickness = 1
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=grid_color, width=thickness)

    # Draw the vertical grid lines
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=thickness)

    return image
