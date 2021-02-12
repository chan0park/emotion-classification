import pandas as pd
import numpy as np
import torch
import pickle
import random
import os
import math
import json
import logging
from tqdm import tqdm, trange

from torch import nn
from torch.nn import BCEWithLogitsLoss, BCELoss, NLLLoss, CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter

from functions import import_data, import_fewshot_data, make_data, data_to_dataloader, copy_file, delete_models, load_best_fewshot_model, load_best_model, load_best_pre_model, load_optimizer, save_model, load_model, load_optimizer_dom
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
from configs import BERT_DIR, BERT_MAX_LENGTH, EMO_EKMAN, EMO_PLUTCHIK, PATH_MODELS, PATH_SUMMARY
from args import args


use_cuda = torch.cuda.is_available() and (not args.no_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
args.n_gpu = torch.cuda.device_count()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

emotions = EMO_PLUTCHIK if args.emotion == "plutchik" else EMO_EKMAN
assert args.target_emotion in emotions, f"target emotion {args.target_emotion} is not in the list of {args.emotion} ({emotions})"

num_labels = 2
args.n_cls = 2
# emo_mapping_i2e = {i: e for i, e in enumerate(emotions)}


def pre_train(model, src_train, src_dev, optimizer):
    if args.verbose:
        logging.info("Pre-training started")
    
    num_step, num_ep, tr_loss, num_tr_examples = 0, 1, 0, 0
    dev_decreased, best_dev_f1 = 0.0, 0.0
    PATIENCE = 4
    F1_THRE = 10
    train_loss_set = []

    len_train = len(src_train)
    src_train_iter = iter(src_train)

    while dev_decreased <= PATIENCE:
        # for step, src_batch in enumerate(src_train):
        try:
            src_batch = next(src_train_iter)
        except StopIteration:
            src_train_iter = iter(src_train)
            src_batch = next(src_train_iter)
            num_ep += 1

        num_step += 1
        batch = tuple(t.to(device) for t in src_batch)
        input_ids, input_mask, labels, token_types = batch

        model.train()   
        optimizer.zero_grad()

        # Forward pass for multilabel classification
        logits_class = model(
            input_ids, token_type_ids=None, attention_mask=input_mask)
        src_loss_class = loss_fn_class(logits_class,labels)

        train_loss_set.append(src_loss_class.item())
        src_loss_class.backward()
        optimizer_step(optimizer)

        tr_loss += src_loss_class.item()
        num_tr_examples += input_ids.size(0)
        if num_step % args.pre_every == 0 and args.verbose:
            logging.info(
                f"Train[p]\t{num_ep: >2}-{num_step: <6}\tSRC-CLS: {tr_loss/num_step:.3f}")
            res, _ = validate(src_dev, 0.5)
            dev_f1 = res[0]
            if dev_f1 >= best_dev_f1:
                dev_decreased = 0
                best_dev_f1 = dev_f1
                if dev_f1 > F1_THRE and args.save:
                    delete_models(f"{args.target_emotion}_pretrained_", args.save_path)
                    logging.info(f"saving trained models to {args.save_path}/{args.target_emotion}_pretrained_{dev_f1:.2f}.pt\n")
                    save_model(args, f"{args.target_emotion}_pretrained_{dev_f1:.2f}.pt", model)
            elif num_step > args.min_step:
                logging.info("dev f1 decreased.")
                dev_decreased += 1

def optimizer_step(opt, bool_clip=True):
    if bool_clip:
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad,
                                            model.parameters()), args.clip)
    opt.step()

def train_src(model, src_train, src_dev, tgt_train, tgt_dev):
    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
    # with an iterator the entire dataset does not need to be loaded into memory
    src_train_dataloader = data_to_dataloader(
        src_train, batch_size=args.batch_size)
    src_dev_dataloader = data_to_dataloader(
        src_dev, batch_size=args.batch_size, bool_valid=True)
    tgt_train_dataloader = data_to_dataloader(
        tgt_train, batch_size=args.batch_size)
    tgt_dev_dataloader = data_to_dataloader(
        tgt_dev, batch_size=args.batch_size, bool_valid=True)

    optimizer_cls, optimizer_dom = load_optimizer(model, args.lr)

    # pre-train until converge
    if not args.skip_pre:
        pre_train(model, src_train_dataloader, src_dev_dataloader, optimizer_cls)
        model = load_best_pre_model(model, args.save_path, (args.n_gpu>1))

    dataloaders = (src_train_dataloader, src_dev_dataloader,
                   tgt_train_dataloader, tgt_dev_dataloader)
    best_tgt_dev_f1, tgt_dev_f1_decreased, PATIENCE = 0.0, 0, 3

def validate(dataloader, threshold=0.5, test=False):
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Variables to gather full output
    val_loss, num_step =  0, 0
    logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []
    pred_doms = []

    # Predict
    for i, batch in enumerate(dataloader):
        num_step += 1
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        input_ids, input_mask, labels, b_token_types = batch
        with torch.no_grad():
            logit_pred = model(input_ids, token_type_ids=None,
                                          attention_mask=input_mask)
            loss_class = loss_fn_class(logit_pred.view(-1, num_labels),labels)
            val_loss += loss_class.item()

            pred_label = torch.sigmoid(logit_pred)
            logit_pred = logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            labels = labels.to('cpu').numpy()
            input_ids = input_ids.to('cpu')
            logit_preds.append(logit_pred)
            true_labels.append(labels)
            pred_labels.append(pred_label)

    val_loss = val_loss/num_step
    # Flatten outputs
    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    # Calculate Accuracy
    # threshold = 0.50
    if threshold is None:
        best_thre, best_f1, best_acc = 0.0, 0.0, 0.0
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            pred_bools = [pl > threshold for pl in pred_labels]
            true_bools = [tl == 1 for tl in true_labels]
            val_f1_accuracy = f1_score(
                true_bools, pred_bools)*100
            val_flat_accuracy = accuracy_score(true_bools, pred_bools)*100
            if val_f1_accuracy > best_f1:
                best_f1 = val_f1_accuracy
                best_acc = val_flat_accuracy
                best_thre = threshold
    else:
        best_thre = threshold
        pred_bools = [pl[1] > pl[0] for pl in pred_labels]
        true_bools = [tl == 1 for tl in true_labels]
        best_f1 = f1_score(true_bools, pred_bools)*100
        best_acc = accuracy_score(true_bools, pred_bools)*100

    if test:
        logging.info(f"\tTest\tF1: {best_f1:.3f}\tAcc: {best_acc:.3f}")
    else:
        logging.info(f"\tValid\tF1: {best_f1:.3f}\tAcc: {best_acc:.3f}")
    return (best_f1, best_acc, val_loss), best_thre
    # return (best_f1, best_acc, best_dom_f1, best_dom_acc, tokenized_texts, pred_labels, true_labels), best_thre

def vec2text_label(labels):
    if labels == 1:
        return args.target_emotion
    else:
        return "neutral"

def write_batch(batch_input_ids, pred_labels, gold_labels, pred_probs):
    sents = [tokenizer.convert_ids_to_tokens(
        sent, skip_special_tokens=True) for sent in batch_input_ids]
    sents = [" ".join(sent) for sent in sents]
    gold_labels = [vec2text_label(label) for label in gold_labels]
    pred_labels = [vec2text_label(label) for label in pred_labels]
    pred_probs = [",".join([str(round(d, 2)) for d in p]) for p in pred_probs]

    with open(args.save_path+f"/{args.target}-test.output", "a") as file:
        for sent, gl, pl, pp in zip(sents, gold_labels, pred_labels, pred_probs):
            file.write(f"{sent}\t{gl}\t{pl}\t{pp}\n")


def eval(test_data, threshold, suffix="", save_report=True, save_output=True):
    test_data = TensorDataset(
        test_data['input_ids'], test_data['attention_masks'], test_data['labels'], test_data['token_type_ids'])
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=args.batch_size)
    model.eval()
    # logit_preds, true_labels, pred_labels, tokenized_texts, pred_bools, true_bools = [], [], [], [], [], []
    logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []
    for i, batch in enumerate(test_dataloader):
        batch_input_ids = batch[0].numpy()
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_token_types = batch
        with torch.no_grad():
            # Forward pass
            b_logit_pred = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask)
            # b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)
            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        if save_output:
        # pred_bools = [pl[1] > pl[0] for pl in pred_labels]
        # true_bools = [tl == 1 for tl in true_labels]
            pred_bool = [pl[1]>pl[0] for pl in pred_label]
            true_bool = [tl==1 for tl in b_labels]
            write_batch(batch_input_ids, pred_bool, true_bool, pred_label)

        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)
    # Flatten outputs
    tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    # Converting flattened binary values to boolean values
    true_bools = [tl == 1 for tl in true_labels]
    # boolean output after thresholding
    # best_thre = threshold if threshold else args.thre
    pred_bools = [pl[1] > pl[0] for pl in pred_labels]

    test_f1_accuracy = f1_score(true_bools, pred_bools)*100
    test_flat_accuracy = accuracy_score(true_bools, pred_bools)*100

    logging.info(
        f"Test\tF1: {test_f1_accuracy:.3f}\tAcc: {test_flat_accuracy:.3f}")

    clf_report = classification_report(
        true_bools, pred_bools, target_names=["non-"+args.target_emotion,args.target_emotion])
    logging.info(clf_report)
    if save_report:
        with open(f'{args.save_path}/classification_report{suffix}.txt', "w") as file:
            file.write(clf_report)
    return test_f1_accuracy, test_flat_accuracy


def train_fewshot(model, tgt_train, tgt_dev, tgt_test, lr=1e-5):
    tgt_train_dataloader = data_to_dataloader(
        tgt_train, batch_size=args.fewshot_batch_size)
    tgt_dev_dataloader = data_to_dataloader(
        tgt_dev, batch_size=args.batch_size, bool_valid=True)
    tgt_test_dataloader = data_to_dataloader(
        tgt_test, batch_size=args.batch_size, bool_valid=True)
    # train until converge
    _train_fewshot(model, tgt_train_dataloader,
                   tgt_dev_dataloader, tgt_test_dataloader, lr)


def _train_fewshot(model, tgt_train, tgt_dev, tgt_test, lr):
    if args.verbose:
        logging.info("Few shot training started")

    optimizer, _ = load_optimizer(model, lr)

    num_step, num_ep, tr_loss, running_tr_loss, num_tr_examples = 0, 1, 0, 0, 0
    dev_decreased, best_dev_f1, last_f1 = 0.0, 0.0, 0.0
    PATIENCE = 3
    train_loss_set = []

    len_train = len(tgt_train)
    tgt_train_iter = iter(tgt_train)

    while dev_decreased < PATIENCE and num_step <= 2000:
        # for step, tgt_batch in enumerate(src_train):
        try:
            tgt_batch = next(tgt_train_iter)
        except StopIteration:
            tgt_train_iter = iter(tgt_train)
            tgt_batch = next(tgt_train_iter)
            num_ep += 1

        num_step += 1
        batch = tuple(t.to(device) for t in tgt_batch)
        input_ids, input_mask, labels, token_types = batch

        model.train()
        optimizer.zero_grad()

        # Forward pass for multilabel classification
        logits_class = model(
            input_ids, token_type_ids=None, attention_mask=input_mask)
        src_loss_class = loss_fn_class(logits_class.view(-1, num_labels),
                                       labels)
        train_loss_set.append(src_loss_class.item())
        src_loss_class.backward()
        optimizer_step(optimizer)

        tr_loss += src_loss_class.item()
        running_tr_loss += src_loss_class.item()
        num_tr_examples += input_ids.size(0)
        if num_step % args.fewshot_every == 0 and args.verbose:
            logging.info(
                f"Train[f]\t{num_ep: >2}-{num_step: <6}\tSRC-CLS: {tr_loss/num_step:.3f}")
            res, _ = validate(tgt_dev, 0.5)
            res_test, _ = validate(tgt_test, 0.5, test=True)
            dev_f1 = res[0]
            test_f1 = res_test[0]
            dev_loss = res[-1]
            if args.tensorboard:
                writer.add_scalar('[f]loss/train', running_tr_loss/args.fewshot_every, num_step)
                writer.add_scalar('[f]loss/valid', dev_loss, num_step)
                writer.add_scalar('[f]f1/valid', dev_f1, num_step)
                writer.add_scalar('[f]f1/test', test_f1, num_step)
            running_tr_loss = 0

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                # if args.save and num_step > 200:
                if args.save:
                    delete_models(f"{args.target_emotion}.pt", args.save_path)
                    logging.info(f"saving trained models to {args.save_path}/{args.target_emotion}.pt\n")
                    save_model(args, f"{args.target_emotion}.pt", model)

            if num_step > 500:
                if dev_f1 > last_f1:
                    dev_decreased = 0
                else:
                    logging.info("dev f1 decreased.")
                    dev_decreased += 1
            last_f1 = dev_f1


# actual code starts from here
# logging setup (tensorboard, log.txt)
logging_handlers = [logging.StreamHandler()]
if args.save:
    logging_handlers += [logging.FileHandler(f'{args.save_path}/log.txt')]
    if args.tensorboard:
        writer = SummaryWriter('runs/'+args.suffix+"_"+args.timestamp[-5:])
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=logging_handlers
)

# load model & tokenizer
model, tokenizer = load_model(args, BERT_DIR, device)
if args.verbose:
    if args.bert_path != None:
        logging.info(
            f"* loaded a bert model from {args.bert_path} and its tokenizer")
        if args.load_from:
            logging.info(
                f"* loaded a bert params from {args.load_from}")
    else:
        logging.info(f"* loaded {args.bert_model} model and its tokenizer")

# load data
if args.verbose:
    logging.info(f"loading data from {args.data_path}")
emo_mapping_e2i = {e: i for i, e in enumerate(emotions)}
target_emotion_idx = emo_mapping_e2i[args.target_emotion]
data, src_data, tgt_data  = import_data(args.data_path, args.source, args.target, args.emotion, target_emotion_idx, tokenizer, args.max_len)
src_train, src_dev, src_test = src_data
tgt_train, tgt_dev, tgt_test = tgt_data


if args.few_shot:
    few_shot_train = import_fewshot_data(data[args.target], args.num_few_shot, target_emotion_idx, tokenizer)

# define loss functions
loss_fn_class = CrossEntropyLoss()
loss_fn_class_noreduce = CrossEntropyLoss(reduction="none")
loss_fn_domain = CrossEntropyLoss()
loss_fn_domain_joint = BCELoss()

# start training
zero_src, zero_tgt, few_src, few_tgt = None, None, None, None
if not args.skip_training:
    train_src(model, src_train, src_dev, tgt_train, tgt_dev)
    model = load_best_model(model, args.save_path, args.load_from, (args.n_gpu>1))

    zero_src = eval(src_test, args.thre, suffix="_zero_src")
    zero_tgt = eval(tgt_test, args.thre, suffix="_zero_tgt")

if args.few_shot:
    train_fewshot(model, few_shot_train, tgt_dev, tgt_test, lr=1e-5)
    model = load_best_fewshot_model(model, args.save_path, lr=1e-5, multi_gpu=(args.n_gpu>1))

    few_src = eval(src_test, args.thre, suffix="_few_src")
    few_tgt = eval(tgt_test, args.thre, suffix="_few_tgt")

def save_final_summary(args, zero_src, zero_tgt, few_src, few_tgt):
    save_path_name = args.save_path.split("/")[-1]
    res_dict = {"bert-model":args.bert_path_model, "emo":args.target_emotion, "seed":args.seed, "src":str(args.source), "tgt":str(args.target), 'loss_ratio': args.loss_ratio, 'num_fewshot':args.num_few_shot}
    for res, zero_few, src_tgt in [(zero_src, "zero", "src"), (zero_tgt, "zero", "tgt"), (few_src, "few", "src"), (few_tgt, "few", "tgt")]:
        f1, acc = res if res is not None else (None, None)
        res_dict[f"{zero_few}_{src_tgt}_f1"] = f1
        res_dict[f"{zero_few}_{src_tgt}_acc"] = acc
    res_dict["savedir"] = args.save_path
    with open(args.save_path+"/summary.json", "w") as file:
        json.dump(res_dict, file)
    with open(args.summary_path+f"{save_path_name}.json", "w") as file:
        json.dump(res_dict, file)

if args.save:
    save_final_summary(args, zero_src, zero_tgt, few_src, few_tgt)
    if args.tensorboard:
        writer.close()