import os
import sys
import json
import pickle
import random

assert len(sys.argv)>=3, "argument missing.\n usage: process_annotations.py PATH_ANNOTATION DIR_SAVE\nGiven args: "+str(sys.argv)
path_annotation = sys.argv[1]
path_save = sys.argv[2]

RATIO_TRAIN=0.8
RATIO_TEST=0.1

EMOTION_SCHEME="ekman"
path_emotion = f"data/emo-mapping/{EMOTION_SCHEME}.txt"

if path_annotation.endswith(".pkl"):
    with open(path_annotation, "rb") as file:
        annots = pickle.load(file)
elif path_annotation.endswith(".json"):
    with open(path_annotation, "r") as file:
        annots = json.load(file)
else:
    raise NotImplementedError


with open(path_emotion,"r") as file:
    EMOTIONS = [l.strip() for l in file.readlines()]

INTENTS=["malicious"]

def process_annot(annot):
    def label_to_vec(labels, dims):
        vec = [0]*len(dims)
        for idx, dim in enumerate(dims):
            if dim in labels:
                vec[idx] = 1
        return vec
    
    annot = set(annot)
    if 'hard-to-tell (stance)' in annot:
        vec_intent = None
    else:
        vec_intent = label_to_vec(annot, INTENTS)
    
    if 'hard-to-tell (emotion)' in annot:
        vec_emotion = None
    else:
        vec_emotion = label_to_vec(annot, EMOTIONS)

    return vec_emotion, vec_intent
    
processed_annots_emotion, processed_annots_intent = [], []
for example in annots:
    text = example["text"]
    annot = example["annotation"]
    annot_emotion, annot_intent = process_annot(annot)
    if not annot_emotion is None:
        processed_annots_emotion.append((text, annot_emotion))
    if not annot_intent is None:
        processed_annots_intent.append((text, annot_intent))

def split_data(data):
    random.shuffle(data)
    num_train = round(len(data)*RATIO_TRAIN)
    num_test = round(len(data)*RATIO_TEST)

    data_train = data[:num_train]
    data_test = data[-num_test:]
    data_dev = data[num_train:-num_test]

    split = {"train":data_train, "test":data_test, "dev":data_dev}
    return split

processed_annots_emotion = split_data(processed_annots_emotion)
processed_annots_intent = split_data(processed_annots_intent)

with open(f"{path_save}/sri-{EMOTION_SCHEME}.pkl", "wb") as file:
    pickle.dump(processed_annots_emotion, file)

with open(f"{path_save}/sri-intent.pkl", "wb") as file:
    pickle.dump(processed_annots_intent, file)