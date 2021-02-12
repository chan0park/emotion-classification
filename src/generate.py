import pandas as pd
import numpy as np
import torch
import pickle
import random
import os
import time
import json
import argparse

from models import BertClassifier
from torch.nn import BCEWithLogitsLoss, BCELoss, NLLLoss, CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax

from preprocess_text import build_preprocess
from functions import process_one_text
from multiprocessing import Pool
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
from transformers import BertTokenizer
from tqdm import tqdm, trange
from configs import PATH_PROCESSED, BERT_DIR, BERT_MAX_LENGTH, EMO_EKMAN, EMO_PLUTCHIK, PATH_MODELS, PATH_SAVE_DIR


def process_text(txt):
    try:
        res = text_preprocessor(txt)
    except:
        print("preprocessing error occured for"+txt)
        res = ""
    return res    

def process_data(entries, bool_already_processed=False):
    texts, labels = zip(*entries)
    if not bool_already_processed:
        texts = [process_text(t) for t in texts]
    encodings = tokenizer.batch_encode_plus(
        texts, max_length=args.max_len, padding=True, truncation=True)  # tokenizer's encoding method
    
    data = {}
    # tokenized and encoded sentences
    data['input_ids'] = torch.tensor(encodings['input_ids'])
    data['token_type_ids'] = torch.tensor(
        encodings['token_type_ids'])  # token type ids
    data['attention_masks'] = torch.tensor(
        encodings['attention_mask'])  # attention masks
    data['labels'] = torch.tensor(labels)
    return data


def write_batch(batch_input_ids, pred_labels, gold_labels, pred_probs):
    def vec2text_label(given_label):
        if given_label == True or given_label == 1:
            return args.target_emotion
        else:
            return "neutral"
    sents = [tokenizer.convert_ids_to_tokens(
        sent, skip_special_tokens=True) for sent in batch_input_ids]
    sents = [" ".join(sent) for sent in sents]
    gold_labels = [vec2text_label(label) for label in gold_labels]
    pred_labels = [vec2text_label(label) for label in pred_labels]
    pred_probs = [",".join([str(round(d, 2)) for d in p]) for p in pred_probs]
    with open(args.save_path, "a") as file:
        for sent, gl, pl, pp in zip(sents, gold_labels, pred_labels, pred_probs):
            file.write(f"{sent}\t{gl}\t{pl}\t{pp}\n")


def generate(test_data, save_report=False, threshold=0.5):
    test_data = TensorDataset(
        test_data['input_ids'], test_data['attention_masks'], test_data['labels'], test_data['token_type_ids'])
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=args.batch_size)

    model.eval()
    logit_preds, true_labels, pred_labels, tokenized_texts, pred_bools, true_bools = [
    ], [], [], [], [], []
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_input_ids = batch[0].numpy()
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels, b_token_types = batch
        with torch.no_grad():
            # Forward pass
            b_logit_pred = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask)
            pred_label = torch.sigmoid(b_logit_pred)
            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        # pred_bool = [[pl > threshold for pl in sublist]
                    #  for sublist in pred_label]
        pred_bool = [pl[1] > threshold for pl in pred_label]
        # true_bool = [[tl == 1 for tl in sublist] for sublist in b_labels]
        true_bool = [tl==1 for tl in b_labels]

        write_batch(batch_input_ids, pred_bool, true_bool, pred_label)
        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)
        # pred_bools.extend(pred_bool)
        # true_bools.extend(true_bool)

def str2vec_label(labels):
    vec_labels = [0] * num_labels
    if labels != "neutral":
        labels = labels.split(",")
        for label in labels:
            label = label.strip()
            vec_labels[emo_mapping_e2i[label]] = 1
    return vec_labels


def import_txt_file(path):
    with open(path, "r") as file:
       raw_lines = [l.strip().split("\t") for l in file.readlines()]
    lines = []
    for l in raw_lines:
        if len(l) == 2:
            text, str_label = l
            label = str2vec_label(str_label)
        elif len(l) == 1:
            text = l[0]
            label = 0
        lines.append((text, label))
    return lines


def import_pickle_file(path):
    with open(path, "rb") as file:
        data = pickle.load(file)
    texts, ids = [], []
    bool_already_processed = False
    for d in data.values():
        if not bool_already_processed:
            if 'processed_text' in d:
                bool_already_processed = True
        if 'processed_text' in d:
            texts.append((d['processed_text'], 0))
        else:
            texts.append((d['full_text'], 0))
        ids.append(str(d['id']))
    return texts, bool_already_processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process arguments for fine-tuning bert.')
    parser.add_argument('--model_path', '-l',
                        action='store', type=str, required=True)
    parser.add_argument('--file_path', '-f', action='store', type=str, required=True,
                        help='file to classify')
    parser.add_argument('--save_path', action='store', type=str, required=True)

    parser.add_argument('--emotion', '-emo', action='store',
                        type=str, default="ekman", choices=["plutchik", "ekman"])
    parser.add_argument('--target_emotion', '-tgtemo', action='store',
                        type=str, default="joy", choices=["disgust","fear","anger","sadness","surprise","joy","disapproval","aggressiveness","optimism","remorse","love","awe","contempt","submission"])

    parser.add_argument('--bert_model', action='store', type=str, default="bert-base-uncased",
                        help='the name of the bert model')
    parser.add_argument('--bert_path', action='store', type=str, default=None,
                        help='the path to the bert model')

    parser.add_argument('--hidden_size', '-hs', action='store', type=int, default=768,
                        help='# of training epochs')
    parser.add_argument('--dp', '-dp', action='store', type=float, default=0.0,
                        help='# of training epochs')
    parser.add_argument('--ddp', '-ddp', action='store', type=float, default=0.0,
                        help='# of training epochs')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=64,
                        help='size of the batch used in training')
    parser.add_argument('--max_len', '-ml', action='store', type=int, default=BERT_MAX_LENGTH,
                        help='size of the batch used in training')
    parser.add_argument('--seed', '-seed',
                        action='store', type=int, default=17)

    parser.add_argument('--verbose', '-v', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--suffix', '-suffix', action='store', type=str, default='',
                        help='suffix for model save_path')
    parser.add_argument('--notes', '-n', action='store', type=str, default='',
                        help='mark notes about the model')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and (not args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.verbose:
        print(f"using device: ", str(device))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # # check if the save file exists
    # if os.path.isfile(args.save_path):
    #     with open(args.save_path, "r") as file:
    #         lines = file.readlines()
    #     if len(lines) > 3:
    #         raise f"{args.save_path} already exists. Ending the job."

    with open(args.save_path, "w") as file:
        file.write("\t".join(["text", "gold", "model_pred", "pred_prob"])+"\n")

    emo_mapping = EMO_PLUTCHIK if args.emotion == "plutchik" else EMO_EKMAN
    emo_mapping_i2e = {i: e for i, e in enumerate(emo_mapping)}
    emo_mapping_e2i = {e: i for i, e in enumerate(emo_mapping)}
    num_labels = 2
    args.n_cls = 2

    if args.bert_path != None:
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_path, do_lower_case=True)  # tokenizer
        model = BertClassifier(args, cache_dir=args.bert_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True, cache_dir=BERT_DIR)  # tokenizer
        model = BertClassifier(args, cache_dir=BERT_DIR)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    if args.verbose:
        print(f"model loaded from {args.model_path}")
    model.to(device)

    text_preprocessor = build_preprocess(demojize=True, textify_emoji=True, mention_limit=3,
                                        punc_limit=3, lower_hashtag=True, segment_hashtag=True, add_cap_sign=True)

    loss_fn_class = BCEWithLogitsLoss()
    loss_fn_domain = CrossEntropyLoss()
    loss_fn_domain_joint = BCELoss()

    if args.file_path.endswith(".pkl"):
        data, bool_already_processed = import_pickle_file(args.file_path)
    else:
        data = import_txt_file(args.file_path)
        bool_already_processed = False

    if args.verbose:
        print("processing the data")
    tgt_test = process_data(data, bool_already_processed)

    if args.verbose:
        print("starting to generate")
    generate(tgt_test)
