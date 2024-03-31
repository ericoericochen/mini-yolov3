import torch
from PIL import Image, ImageDraw
import torchvision
from typing import Union
import matplotlib.pyplot as plt
from typing import Literal


def coco_to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
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
    Converts COCO format (x_min, y_min, w, h) to (x_c, y_c, w, h) format

    Params:
        - bbox: (B, 4) in COCO format (x_min, y_min, width, height)
    """
    # get center of bounding box
    x = bbox[:, 0] + bbox[:, 2] / 2
    y = bbox[:, 1] + bbox[:, 3] / 2

    bbox = torch.stack([x, y, bbox[:, 2], bbox[:, 3]], dim=1)
    return bbox


def xywh_to_xyxy(bbox: torch.Tensor):
    """
    Converts (x_c, y_c, w, h) format to (x_min, y_min, x_max, y_max) format

    Params:
        - bbox: (B, 4) in (x, y, w, h) format
    """

    # get top left coordinates
    x1 = bbox[:, 0] - bbox[:, 2] / 2
    y1 = bbox[:, 1] - bbox[:, 3] / 2

    # get bottom right coordinates
    x2 = bbox[:, 0] + bbox[:, 2] / 2
    y2 = bbox[:, 1] + bbox[:, 3] / 2

    bbox = torch.stack([x1, y1, x2, y2], dim=1)
    return bbox


def xyxy_to_xywh(bbox: torch.Tensor):
    """
    Converts (x_min, y_min, x_max, y_max) format to (x_c, y_c, w, h) format

    Params:
        - bbox: (B, 4) in (x_min, y_min, x_max, y_max) format
    """

    # get center of bounding box
    x = (bbox[:, 0] + bbox[:, 2]) / 2
    y = (bbox[:, 1] + bbox[:, 3]) / 2

    # get width and height
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]

    bbox = torch.stack([x, y, w, h], dim=1)
    return bbox


def draw_bounding_boxes(
    image: Image.Image,
    bbox: torch.Tensor,
    labels: Union[torch.Tensor, list[str]],
    format: Literal["coco", "xyxy", "xywh"] = "coco",
):
    """
    Draw bounding boxes on an image with labels

    Params:
        - image: image to draw bounding boxes on
        - bbox: normalized bbox between [0, 1] (B, 4)
        - labels: list of labels for each bounding box
        - format: format of the bounding box, either "coco" | "xyxy" | "xywh"
    """
    if isinstance(labels, torch.Tensor):
        assert len(labels.shape) == 1
        labels = [str(label) for label in labels.tolist()]

    # convert image to be in range [0, 255]
    image = (image * 255).to(torch.uint8)

    # convert bbox to xyxy
    bbox = bbox.clone()
    if format == "coco":
        bbox = coco_to_xyxy(bbox)
    elif format == "xywh":
        bbox = xywh_to_xyxy(bbox)

    # rescale bbox to image size
    w, h = image.shape[2], image.shape[1]

    bbox[:, [0, 2]] *= w  # scale x coords
    bbox[:, [1, 3]] *= h  # scale y coords

    bounding_box_image = torchvision.utils.draw_bounding_boxes(
        image, boxes=bbox, labels=labels
    )
    pil_image = torchvision.transforms.ToPILImage()(bounding_box_image)

    return pil_image


def to_tensor(image: Image.Image):
    """
    Convert image to tensor
    """
    return torchvision.transforms.ToTensor()(image)


def to_pil_image(image: torch.Tensor):
    """
    Convert tensor to image
    """
    return torchvision.transforms.ToPILImage()(image)


def draw_grid(image: Image.Image, grid_size: int):
    """
    Draw a grid on an image. The total number of cells is `grid_size x grid_size`.
    """

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


def box_iou(
    bbox1: torch.Tensor,
    bbox2: torch.Tensor,
    format: Literal["coco", "xyxy", "xywh"] = "xyxy",
) -> torch.Tensor:
    """
    Calculates the IOU between bounding boxes.
    """

    # convert bbox to xyxy
    if format == "coco":
        bbox1 = coco_to_xyxy(bbox1)
        bbox2 = coco_to_xyxy(bbox2)
    elif format == "xywh":
        bbox1 = xywh_to_xyxy(bbox1)
        bbox2 = xywh_to_xyxy(bbox2)

    # calculate IOU
    x1_min, y1_min, x1_max, y1_max = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    x2_min, y2_min, x2_max, y2_max = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    a = torch.max(x1_min, x2_min)
    b = torch.max(y1_min, y2_min)
    c = torch.min(x1_max, x2_max)
    d = torch.min(y1_max, y2_max)

    intersection = torch.relu(c - a) * torch.relu(d - b)
    union = (
        (x1_max - x1_min) * (y1_max - y1_min)
        + (x2_max - x2_min) * (y2_max - y2_min)
        - intersection
    )

    iou = intersection / union

    return iou


# def scale_bbox_to_global_xy(bboxes: torch.Tensor) -> torch.Tensor:
#     """
#     Scales coordinates (x, y) of bounding box predictions to global image coordinates.

#     Bounding Box predictions are relative to the cell they're in.

#     Expects bbox to be in (x_c, y_c, w, h) format
#     """

#     cell_size = 1 / grid_size
#     x_indices = list(range(0, dim, num_classes + 5))
#     x_offsets = torch.arange(0, grid_size).repeat(grid_size, 1) * cell_size  # (H, W)

#     y_indices = list(range(1, dim, num_classes + 5))
#     y_offsets = x_offsets.T

#     pred[:, x_indices, :, :] += x_offsets
#     pred[:, y_indices, :, :] += y_offsets
