#!/bin/bash
poetry run python3 train_svhn.py \
    --model_config=../configs/experiment.json \
    --image_size=32 \
    --batch_size=64 \
    --num_epochs=200 \
    --lr=1e-3 \
    --weight_decay=0.00001 \
    --data_augment \
    --augment_prob=0.2 \
    --eval_every=5 \
    --checkpoint_epoch=10 \
    --save_dir=../experiments/fatass \
    # --save_dir=../checkpoints/svhn_experiment_small_wd \
    # --save_dir=../checkpoints/svhn_experiment_big \