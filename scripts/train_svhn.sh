#!/bin/bash
poetry run python3 train_svhn.py \
    --model_config=../configs/experiment.json \
    --image_size=64 \
    --batch_size=128 \
    --num_epochs=500 \
    --lr=8e-4 \
    --weight_decay=0.0005 \
    --data_augment \
    --augment_prob=0.05 \
    --eval_every=20 \
    --checkpoint_epoch=10 \
    --save_dir=../checkpoints/awesome_whoo \
    # --save_dir=../checkpoints/svhn_experiment_small_wd \
    # --save_dir=../checkpoints/svhn_experiment_big \