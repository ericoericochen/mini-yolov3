#!/bin/bash
poetry run python3 train_oxford_pet.py \
    --data_dir=../data/oxford-iiit-pet \
    --model_config=../configs/experiment.json \
    --image_size=128 \
    --batch_size=64 \
    --num_epochs=300 \
    --lr=1e-3 \
    --weight_decay=0.1 \
    --data_augment \
    --lambda_coord=5.0 \
    --lambda_noobj=0.5 \
    --augment_prob=0.0 \
    --eval_every=10 \
    --checkpoint_epoch=10 \
    --save_dir=../experiments_new/final3 \
