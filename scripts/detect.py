import argparse

import sys

sys.path.append("../")

from PIL import Image

import os
from typing import Callable, Union

import PIL.Image
import PIL.ImageOps
import requests
import json
from mini_yolov3.model import MiniYoloV3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="../configs/smol.json")
    parser.add_argument("--weights_path", type=str)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

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

    model = MiniYoloV3.from_config(model_config)
    image = load_image(args.image_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
