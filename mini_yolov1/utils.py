import torch
from PIL import Image, ImageDraw
import torchvision
from typing import Union


def draw_bounding_boxes(
    image: Image.Image,
    bboxes: torch.Tensor,
    labels: Union[torch.Tensor, list[str]] = None,
):
    """
    Draw bounding boxes on an image with labels

    Params:
        - image: image to draw bounding boxes on
        - bbox: normalized bbox between [0, 1] (B, 4), MUST be in xyxy format
        - labels: list of labels for each bounding box
        - format: format of the bounding box, either "coco" | "xyxy" | "xywh"
    """
    if isinstance(labels, torch.Tensor):
        assert len(labels.shape) == 1
        labels = [str(label) for label in labels.tolist()]

    # convert image to be in range [0, 255]
    image = (image * 255).to(torch.uint8)

    # rescale bbox to image size
    w, h = image.shape[2], image.shape[1]

    bboxes = bboxes.clone()
    bboxes[:, [0, 2]] *= w  # scale x coords
    bboxes[:, [1, 3]] *= h  # scale y coords

    bounding_box_image = torchvision.utils.draw_bounding_boxes(
        image, boxes=bboxes, labels=labels
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


def draw_grid(image: Union[Image.Image, torch.Tensor], grid_size: int):
    """
    Draw a grid on an image. The total number of cells is `grid_size x grid_size`.
    """
    if isinstance(image, torch.Tensor):
        image = to_pil_image(image)

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


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters())
