#!/bin/bash

# This are the hyperparameters you can change to fit your data
python train.py --data_dir=./quick_draw \
--n_dim=16 \
--num_layers 2 \
--image_size=28 \
--image_depth=1 \
--filters=8 \
--learning_rate=0.0005 \
--decay_rate=0.01 \
--batch_size=128 \
--epochs=30
