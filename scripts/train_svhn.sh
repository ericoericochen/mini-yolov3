#!/bin/bash
poetry run python3 train_svhn.py \
    --model_config=../configs/experiment.json \
    --image_size=32 \
    --batch_size=64 \
    --num_epochs=4 \
    --lr=3e-4 \
    --weight_decay=0.0 \
    --eval_every=5 \
    --save_dir=../checkpoints/svhn \