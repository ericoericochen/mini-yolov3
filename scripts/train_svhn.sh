#!/bin/bash
poetry run python3 train_svhn.py \
    --model_config=../configs/experiment.json \
    --image_size=32 \
    --batch_size=64 \
    --num_epochs=25 \
    --lr=1e-2 \
    --weight_decay=0.0 \
    --eval_every=1000 \
    --save_dir=../checkpoints/svhn_experiment2 \
    --checkpoint_epoch=100000 \