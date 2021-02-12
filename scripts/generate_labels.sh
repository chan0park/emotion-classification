#!/bin/bash

for emotion in joy surprise sadness anger disgust fear
do
    echo $emotion
    python src/generate.py --model_path saved_models/hur2go/"$emotion".pt --file_path test.txt --save_path res/test."$emotion" --target_emotion $emotion
done