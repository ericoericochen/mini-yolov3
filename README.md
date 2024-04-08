# YoloV3 Smol

A smol implementation of "YOLOv3: An Incremental Improvement" trained on the Street View House Number (SVNH) dataset from Huggingface.

Paper Link: https://arxiv.org/abs/1804.02767

Dataset: https://huggingface.co/datasets/svhn

## Intro

What are the objects in an image and their bounding boxes?

YOLO V3 is a neural net created by Joseph Redmon and Ali Farhadi for object detection. The network takes in an image and predicts a set of bounding boxes, confidence scores, and class probabilities.

It is fast and can be optimized end to end without additional networks such as Region Proposal Network (RPN) used in Masked R-CNNs.

### Model

Convnet backbone extracts latent features from the image which can be decoded by "heads" to get bounding box, confidence scores, and class probabiliy predictions. To better detect small vs. big objects, YOLO makes predictions at different scales.

![](https://viso.ai/wp-content/uploads/2021/02/YOLOv3-how-it-works.jpg)
_Diagram of YOLO V3. ConvNet backbone (DarkNet 53) with residual connections followed by upsampling blocks and heads to predict bounding boxes at the different resolutions._

### Our Smol Implementation

We made several changes to YOLO V3 to "smolify" the model. We made the Darknet backbone significantly smaller, used 2 prediction heads instead of 3, and for each prediction head used only 1 anchor.

## Installation

We use poetry :). Here for more info: https://python-poetry.org/.

```bash
poetry init
poetry shell
poetry install
```

## Quickstart

Scripts to train YOLO V3 smol and run inference. Make sure you're in the `./scripts` directory before running any of the commands below.

### Download SVHN

### Training

```bash
poetry run python3 train_svhn.py \
    --model_config="../configs/smol.json" \
    --image_size=32 \
    --batch_size=64 \
    --num_epochs=100 \
    --lr=1e-3 \
    --weight_decay=0.0001 \
    --data_augment \ # turn on color jitter
    --augment_prob=0.05 \ # color jitter probability
    --eval_every=10 \ # run mAP eval every <eval_every> epoch
    --checkpoint_epoch=10 \ # save weights every <checkpoint_epoch> epoch
    --save_dir="../checkpoints/yolov3_run" # save dir for training run
```

or copy paste the above in `./scripts/train_svhn.sh` and run

```bash
./train_svhn.sh
```

### Inference

```bash
poetry run python3 detect.py \
    --model_config="../configs/smol.json" \
    --weights="../weights/weights.pt" \
    --confidence_threshold=0.5 \
    --image_path="../examples/nine_two.jpg" \
    --save_dir="../results"
```

or copy paste the above in `./scripts/detect.sh` and run

```bash
./detect.sh
```

## Config

Model Configs

- `num_classes`: number of classes
- `image_size`: input image size
- `backbone_layers`: backbone architecture

  - `type`: `"Downsample"` | `"ResidualBlock"`
    - `"Downsample"`: conv layer that downsamples spatial resolution by 2x
    - `"ResidualBlock"`: conv layer with residual connection
  - If `type="Downsample"`, specify `in_channels` and `out_channels`
  - If `type="ResidualBlock"`, specify `num_blocks` and `channels`

- `anchors`: 2d list list defining the width and height of anchor boxes. The number of anchor boxes must equal `num_anchors_per_scale * num_detection_layers`. Works best if they are in descending order.
- `num_anchors_per_scale`: number of anchors for each detection layer
- `num_detection_layers`: number of yolo detection layers

### Example

```json
// config/smol.json
{
  "num_classes": 10,
  "image_size": 32,
  "backbone_layers": [
    {
      "type": "Downsample",
      "in_channels": 3,
      "out_channels": 16
    },
    {
      "type": "ResidualBlock",
      "num_blocks": 2,
      "channels": 16
    },
    {
      "type": "Downsample",
      "in_channels": 16,
      "out_channels": 32
    },
    {
      "type": "ResidualBlock",
      "num_blocks": 4,
      "channels": 32
    },
    {
      "type": "Downsample",
      "in_channels": 32,
      "out_channels": 64
    }
  ],
  "anchors": [
    [0.15, 0.73],
    [0.1, 0.46]
  ],
  "num_anchors_per_scale": 1,
  "num_detection_layers": 2
}
```

## Benchmarks
