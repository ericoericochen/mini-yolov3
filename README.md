# YoloV3 Mini

A smol implementation of "YOLOv3: An Incremental Improvement" trained on the Street View House Number (SVNH) dataset from Huggingface.

Paper Link: https://arxiv.org/abs/1804.02767

Dataset: https://huggingface.co/datasets/svhn

## Intro

What are the objects in an image and what are their bounding boxes?

YOLO V3 is a neural net created by Joseph Redmon and Ali Farhadi for object detection. The network takes in an image and predicts a set of bounding boxes, confidence scores, and class probabilities.

It is fast and can be optimized end to end without additional networks such as Region Proposal Network (RPN) used in Masked R-CNNs.

### Model TLDR

Convnet backbone extracts latent features from the image which can be decoded by "heads" to get bounding box, confidence scores, and class probabilities predictions.

![](https://viso.ai/wp-content/uploads/2021/02/YOLOv3-how-it-works.jpg)
_Diagram of YOLO V3. ConvNet backbone (DarkNet 53) with residual connections followed by upsampling blocks and heads to predict bounding boxes at the different resolutions._

![](https://miro.medium.com/v2/resize:fit:574/1*15uBgdR3_rNZzx665Leang.jpeg)
_From the YOLO V1 Paper. YOLO divides the image into patches and makes bounding box predictions for each patch._

![](https://www.researchgate.net/publication/345398664/figure/fig3/AS:1023312901709828@1620988216723/YOLOv3-bounding-box-calculation.png)
_Transformation of the network's bounding box predictions $(t_x, t_y, t_w, t_h)$ into global coordinates on the image $(b_x, b_y, b_w, b_h)$._

## Quickstart

## Training

## Benchmarks
