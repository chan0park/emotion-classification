#!/bin/bash
emotion=$1

python src/train.py -v -s --source hurricane blm --target go --suffix h2g_"$emotion" --few_shot --seed $seed --target_emotion $emotion --model_path saved_models/hur2go/ --summary_path model_summary/hur2go/