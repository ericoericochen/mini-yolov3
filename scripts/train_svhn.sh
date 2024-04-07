#!/bin/bash
poetry run python3 train_svhn.py \
    --model_config=../configs/experiment2.json \
    --image_size=32 \
    --batch_size=64 \
    --num_epochs=100 \
    --lr=3e-4 \
    --weight_decay=0.01 \
    --data_augment \
    --augment_prob=0.05 \
    --eval_every=10 \
    --save_dir=../checkpoints/svhn_experiment_lol \
    --checkpoint_epoch=10 \
    # --save_dir=../checkpoints/svhn_experiment_big \