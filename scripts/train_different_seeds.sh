#!/bin/bash

source /projects/tir1/users/chanyoun/anaconda2/etc/profile.d/conda.sh
conda activate blm
emotion=$1
target_data=$2

for seed in 17 15 19 23 29
do
    python src/train.py -v -s --source hurricane go --target "$target_data" --suffix $seed --few_shot --seed $seed --target_emotion $emotion --model_path saved_models/sri/ --summary_path model_summary/sri/
done
