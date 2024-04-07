#!/bin/bash
poetry run python3 train_svhn.py \
    --model_config=../configs/experiment2.json \
    --image_size=32 \
    --batch_size=64 \
    --num_epochs=50 \
    --lr=1e-3 \
    --weight_decay=0.01 \
    --data_augment \
    --augment_prob=0.1 \
    --eval_every=10 \
    --save_dir=../checkpoints/svhn_experiment_big \
    --checkpoint_epoch=10 \