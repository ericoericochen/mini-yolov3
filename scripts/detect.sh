#!/bin/bash
poetry run python3 detect.py \
    --model_config="../configs/experiment.json" \
    --weights="../checkpoints/svhn_experiment/weights.pt" \
    --confidence_threshold=0.4 \
    --image_path="../examples/nine_two.jpg" \
    --save_dir="../results" \