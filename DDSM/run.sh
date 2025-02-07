#!/bin/bash
set -e
date_time=$(date "+%y%m%d_%H%M%S")
python3 train.py -c ./config/train_config_top5.json -gpu 0  -id ${date_time}_140_0 -seed 140 &

