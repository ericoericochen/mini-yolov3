#!/bin/bash
poetry run python3 train_svhn.py \
    --model_config=../configs/experiment.json \
    --image_size=64 \
    --batch_size=64 \
    --num_epochs=500 \
    --lr=1e-3 \
    --weight_decay=0.0001 \
    --data_augment \
    --augment_prob=0.05 \
    --eval_every=20 \
    --checkpoint_epoch=10 \
    --save_dir=../checkpoints/awesome \
    # --save_dir=../checkpoints/svhn_experiment_small_wd \
    # --save_dir=../checkpoints/svhn_experiment_big \