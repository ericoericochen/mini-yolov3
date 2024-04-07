import argparse


from PIL import Image

import os
from typing import Callable, Union

import PIL.Image
import PIL.ImageOps
import requests
import json
import sys
import torch
from torchvision.ops import box_convert
import matplotlib.pyplot as plt

sys.path.append("../")
from mini_yolov3.model import MiniYoloV3
from mini_yolov3.utils import to_tensor, draw_bounding_boxes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config", type=str, default="../configs/experiment.json"
    )
    parser.add_argument("--weights_path", type=str, required=False)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    return parser.parse_args()


# https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/loading_utils.py
def load_image(
    image: Union[str, PIL.Image.Image],
    convert_method: Callable[[PIL.Image.Image], PIL.Image.Image] = None,
) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], optional):
            A conversion method to apply to the image after loading it. When set to `None` the image will be converted
            "RGB".

    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = PIL.ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image


def main(args):
    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    model = MiniYoloV3(**model_config)
    image = load_image(args.image_path)

    resized_image = image.resize((model.image_size, model.image_size), Image.LANCZOS)
    inference_image = to_tensor(resized_image)

    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path))

    print(inference_image.shape)

    model.eval()
    output = model.inference(
        inference_image.unsqueeze(0), confidence_threshold=args.confidence_threshold
    )
    bounding_boxes = output.bboxes[0]

    bbox = draw_bounding_boxes(
        to_tensor(image),
        box_convert(
            bounding_boxes["bboxes"],
            in_fmt="cxcywh",
            out_fmt="xyxy",
        ),
        bounding_boxes["labels"],
    )

    plt.imshow(bbox)
    save_path = os.path.join(args.save_dir, os.path.basename(args.image_path))
    plt.savefig(save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
