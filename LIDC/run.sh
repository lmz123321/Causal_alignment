#!/bin/bash
set -e
seed=126
date_time=$(date "+%y%m%d_%H%M%S")
python3 train.py -c ./config/train_config_top5.json -gpu 5  -id ${date_time}_126_0 -seed 126