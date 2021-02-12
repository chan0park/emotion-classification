#!/bin/bash

source /projects/tir1/users/chanyoun/anaconda2/etc/profile.d/conda.sh
conda activate blm
emotion=$1

for seed in 17 15 19 23 29
do
    python src/train.py -v -s --source hurricane --target go --suffix $seed --few_shot --seed $seed --target_emotion $emotion --model_path saved_models/hur2go/ --summary_path model_summary/hur2go/
done
