#!/bin/bash
poetry run python3 train_svhn.py \
    --model_config=../configs/experiment.json \
    --image_size=128 \
    --batch_size=64 \
    --num_epochs=100 \
    --lr=1e-3 \
    --weight_decay=0.0001 \
    --data_augment \
    --augment_prob=0.05 \
    --eval_every=5 \
    --checkpoint_epoch=10 \
    --save_dir=../experiments/lion_60k \
