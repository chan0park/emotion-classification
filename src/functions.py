import os
import torch
import pickle
import shutil
import json
import random
from torch.nn.functional import pad
from transformers import AdamW, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from models import BertClassifier
from preprocess_text import build_preprocess
from configs import AVAILABLE_DATA

def process_one_text(text_preprocessor, txt):
    try:
        res = text_preprocessor(txt)
    except:
        print("preprocessing error occured for"+t)
        res = ""
    return res

def process_data(entries, tokenizer, text_preprocessor, target_emotion_idx, max_len=256):
    texts, labels = zip(*entries)
    labels = [lbl[target_emotion_idx] for lbl in labels]
    texts = [process_one_text(text_preprocessor, t) for t in texts]

    encodings = tokenizer.batch_encode_plus(
        texts, max_length=max_len, padding=True, truncation=True)  # tokenizer's encoding method

    data = make_data(encodings['input_ids'], encodings['token_type_ids'], encodings['attention_mask'], labels)
    return data

def import_data(data_path, data_source, data_target, emotion, target_emotion_idx, tokenizer, max_len=256):
    data = {}
    for name in AVAILABLE_DATA:
        with open(data_path+f"{name}-{emotion}.pkl", "rb") as file:
            data[name] = pickle.load(file)
    # with open(data_path+f"go-{emotion}.pkl", "rb") as file:
    #     data["go"] = pickle.load(file)
    # with open(data_path+f"hurricane-{emotion}.pkl", "rb") as file:
    #     data["hurricane"] = pickle.load(file)
    # with open(data_path+f"blm-{emotion}.pkl", "rb") as file:
    #     data["blm"] = pickle.load(file)

    text_preprocessor = build_preprocess(demojize=True, textify_emoji=True, mention_limit=3,
                                        punc_limit=3, lower_hashtag=True, segment_hashtag=True, add_cap_sign=True)

    src_train_data, src_dev_data, src_test_data = [], [], []
    for src in data_source:
        src_train_data += data[src]["train"]
        src_dev_data += data[src]["dev"]
        src_test_data += data[src]["test"]

    src_train = process_data(src_train_data, tokenizer, text_preprocessor, target_emotion_idx, max_len)
    src_dev = process_data(src_dev_data, tokenizer, text_preprocessor, target_emotion_idx, max_len)
    src_test = process_data(src_test_data, tokenizer, text_preprocessor, target_emotion_idx, max_len)

    tgt_train = process_data(data[data_target]["train"], tokenizer, text_preprocessor, target_emotion_idx, max_len)
    tgt_dev = process_data(data[data_target]["dev"], tokenizer, text_preprocessor, target_emotion_idx, max_len)
    tgt_test = process_data(data[data_target]["test"], tokenizer, text_preprocessor, target_emotion_idx, max_len)

    return data, (src_train, src_dev, src_test), (tgt_train, tgt_dev, tgt_test)

def import_fewshot_data(data_target, num_few_shot, target_emotion_idx, tokenizer):
    if num_few_shot == 0:
        return None
    
    text_preprocessor = build_preprocess(demojize=True, textify_emoji=True, mention_limit=3,
                                        punc_limit=3, lower_hashtag=True, segment_hashtag=True, add_cap_sign=True)
    if num_few_shot>0:
        if num_few_shot >= len(data_target["train"]):
            few_shot_train = process_data(data_target["train"], tokenizer, text_preprocessor, target_emotion_idx)
        else:
            if "fewshot-" + str(num_few_shot) in data_target:
                # assert "fewshot-" + \
                #     str(num_few_shot) in data_target, f"fewshot-{num_few_shot} not in the target data"
                few_shot_train = process_data(
                    data_target["fewshot-"+str(num_few_shot)], tokenizer, text_preprocessor, target_emotion_idx)
            else:
                random_sampled_data = random.sample(data_target["train"], num_few_shot)
                few_shot_train = process_data(random_sampled_data, tokenizer, text_preprocessor, target_emotion_idx)
    elif num_few_shot == -1:
        few_shot_train = process_data(data_target["train"], tokenizer, text_preprocessor, target_emotion_idx)
    return few_shot_train

def make_data(input_ids, type_ids, masks, labels, probs=None):
    data = {}
    # tokenized and encoded sentences
    data['input_ids'] = torch.tensor(input_ids)
    data['token_type_ids'] = torch.tensor(type_ids)  # token type ids
    data['attention_masks'] = torch.tensor(masks)  # attention masks
    data['labels'] = torch.LongTensor(labels)
    if probs != None:
        data['probs'] = torch.tensor(probs)
    return data

def combine_data(data1, data2, pad_token_id):
    new_data = {}
    for name in ['input_ids', 'token_type_ids', 'attention_masks', 'labels']:
        dim1 = data1[name].shape[1]
        dim2 = data2[name].shape[1]
        if  dim1 > dim2:
            data2[name] = pad(input=data2[name], pad=(0,dim1-dim2), mode="constant", value=pad_token_id)
        elif dim1 < dim2:
            data1[name] = pad(input=data1[name], pad=(0,dim2-dim1), mode="constant", value=pad_token_id)
        new_data[name] = torch.cat([data1[name],data2[name]], axis=0)
    return new_data

def data_to_dataloader(data, batch_size, bool_valid=False):
    if len(data) == 4:
        data = TensorDataset(
            data['input_ids'], data['attention_masks'], data['labels'], data['token_type_ids'])
    elif len(data) == 5:
        data = TensorDataset(
            data['input_ids'], data['attention_masks'], data['labels'], data['token_type_ids'], data['probs'])
    sampler = SequentialSampler(
        data) if bool_valid else RandomSampler(data)
    dataloader = DataLoader(
        data, sampler=sampler, batch_size=batch_size)
    return dataloader


def copy_file(save_path, string_to_include, save_filename):
    file_names = [f for f in os.listdir(save_path) if string_to_include in f]
    assert len(file_names) == 2, "there should be two files (.txt and .pt)"
    for file_name in file_names:
        file_extension = file_name.split(".")[-1]
        shutil.copyfile(f'{save_path}/{file_name}', f'{save_path}/{save_filename}.{file_extension}')

def delete_models(name, save_path):
    for file_name in os.listdir(save_path):
        if file_name.startswith(name):
            os.remove(f"{save_path}/{file_name}")

def load_best_fewshot_model(model, save_path, lr=1e-5, multi_gpu=False):
    file_path = [path for path in os.listdir(
        save_path) if f"fewshot_{lr}_" in path and ".pt" in path]
    assert len(file_path) == 1
    file_path = os.path.join(save_path, file_path[0])
    model = load_model_from_path(model, file_path, multi_gpu)
    return model

def load_model_from_path(model, file_path, multi_gpu=False):
    if multi_gpu > 1:
        model.module.load_state_dict(torch.load(file_path))
    else:
        model.load_state_dict(torch.load(file_path))
    return model


def load_best_model(model, save_path, load_from, multi_gpu=False):
    file_path = [path for path in os.listdir(save_path) if "best_" in path and ".pt" in path]
    if len(file_path) == 0:
        file_path = [path for path in os.listdir(save_path) if "pretrained_" in path and ".pt" in path]
        if len(file_path) == 0:
            model = load_initial_model(model, load_from)
            return model
    assert len(file_path) == 1
    file_path = file_path[0]
    model = load_model_from_path(model, os.path.join(save_path, file_path), multi_gpu)
    return model


def load_best_pre_model(model, save_path, multi_gpu=False):
    file_path = [path for path in os.listdir(save_path) if "pretrained_" in path and ".pt" in path]
    file_path = file_path[0]
    model = load_model_from_path(model, os.path.join(save_path, file_path), multi_gpu)
    return model

class MultipleOptimizer(object):
    def __init__(*op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

def load_optimizer_dom(model, lr):
    optimizer_dom = AdamW(filter(lambda p: p.requires_grad,
                            [p for n, p in model.named_parameters() if "domain" in n]), lr=lr, correct_bias=True)
    return optimizer_dom

def load_optimizer(model, lr):
    optimizer_clf = AdamW(filter(lambda p: p.requires_grad,
                             [p for n, p in model.named_parameters() if not "domain" in n]), lr=lr, correct_bias=True)
    try:
        optimizer_dom = AdamW(filter(lambda p: p.requires_grad,
                                [p for n, p in model.named_parameters() if "domain" in n]), lr=lr, correct_bias=True)
    except:
        optimizer_dom = None
    return optimizer_clf, optimizer_dom

def save_model(args, name, model):
    # logging.info(f"saving trained models to {args.save_path}/{name}\n")
    if args.n_gpu > 1:
        torch.save(model.module.state_dict(),
                   os.path.join(args.save_path, name))
    else:
        torch.save(model.state_dict(), os.path.join(args.save_path, name))
    with open(os.path.join(args.save_path, name.replace(".pt", ".json")), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def load_initial_model(model, load_from):
    if load_from:
        try:
            model.load_state_dict(torch.load(load_from))
        except:
            model.module.load_state_dict(torch.load(load_from))
    else:
        try:
            model.initialize_bert()
        except:
            model.module.initialize_bert()
    return model

def load_model(args, bert_dir, device):
    model = BertClassifier(args, cache_dir=bert_dir)
    model = load_initial_model(model, args.load_from)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    if args.bert_path != None:
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_path, do_lower_case=True)  # tokenizer
    else:
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True, cache_dir=bert_dir)  # tokenizer
    return model, tokenizer