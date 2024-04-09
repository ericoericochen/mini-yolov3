#!/bin/bash
poetry run python3 detect.py \
    --model_config="../configs/test.json" \
    --weights="../checkpoints/newest/weights.pt" \
    --confidence_threshold=0.5 \
    --image_path="../examples/three_six.jpg" \
    --save_dir="../results" \