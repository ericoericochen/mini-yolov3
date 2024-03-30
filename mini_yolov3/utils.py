import torch
from PIL import Image
import torchvision
from typing import Union


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
