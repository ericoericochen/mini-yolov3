#!/bin/bash
poetry run python3 detect.py \
    --model_config="../configs/experiment.json" \
    --weights="../checkpoints/smol/checkpoints/weights_9.pt" \
    --confidence_threshold=0.5 \
    --image_path="../examples/nine_two.jpg" \
    --save_dir="../results" \