import os
import argparse
import time
from datetime import datetime
from configs import PATH_PROCESSED, BERT_DIR, BERT_MAX_LENGTH, EMO_EKMAN, EMO_PLUTCHIK, PATH_MODELS, PATH_BERT_BLM, PATH_BERT_BLM_ONLY, PATH_BERT_HUR, PATH_BERT_HUR_ONLY, PATH_SUMMARY, AVAILABLE_DATA

parser = argparse.ArgumentParser(
    description='Process arguments for fine-tuning bert.')
parser.add_argument('--emotion', '-emo', action='store',
                    type=str, default="ekman", choices=["plutchik", "ekman"])
parser.add_argument('--n_cls', '-n_cls', action='store',
                    type=int, default=None)
parser.add_argument('--source', '-src', nargs='*', action='store',
                    type=str, default=["go"], choices=AVAILABLE_DATA)
parser.add_argument('--target', '-tgt', action='store',
                    type=str, default="hurricane", choices=AVAILABLE_DATA)
parser.add_argument('--target_emotion', '-tgtemo', action='store',
                    type=str, default="joy", choices=["disgust","fear","anger","sadness","surprise","joy","disapproval","aggressiveness","optimism","remorse","love","awe","contempt","submission"])

parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--save_path', action='store', type=str, default=None)
parser.add_argument('--model_path', action='store', type=str, default=PATH_MODELS)
parser.add_argument('--summary_path', action='store', type=str, default=PATH_SUMMARY)
parser.add_argument('--data_path', action='store', type=str, default=PATH_PROCESSED)
parser.add_argument('--load_from', '-l',
                    action='store', type=str, default=None)

parser.add_argument('--seed', '-seed',
                    action='store', type=int, default=17)
parser.add_argument('--verbose', '-v', action='store_true', default=False)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--tensorboard', '-tb', action='store_true', default=False)
parser.add_argument('--append_timestamp', action='store_true', default=False)
parser.add_argument('--timestamp', action='store', type=str, default="")

parser.add_argument('--bert_model', action='store', type=str, default="bert-base-uncased",
                    help='the name of the bert model')
parser.add_argument('--bert_path_model', action='store', type=str, default=None, choices=[None,"blmonly","blm","hur","huronly"])
parser.add_argument('--bert_path', action='store', type=str, default=None,
                    help='the path to the bert model')
parser.add_argument('--suffix', '-suffix', action='store', type=str, default='',
                    help='suffix for model save_path')
parser.add_argument('--notes', '-n', action='store', type=str, default='',
                    help='mark notes about the model')

parser.add_argument('--batch_size', '-bs', action='store', type=int, default=32,
                    help='size of the batch used in training')
parser.add_argument('--fewshot_batch_size', '-fbs', action='store', type=int, default=64,
                    help='size of the batch used in training')
parser.add_argument('--max_len', '-ml', action='store', type=int, default=BERT_MAX_LENGTH,
                    help='size of the batch used in training')

parser.add_argument('--lr', '-lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--fewshot_lr', '-flr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--loss_ratio', '-lra', type=float, default=0.05,
                    help='loss ratio (lr_cls vs. lr_domain)')
parser.add_argument('--clip', type=float, default=5,
                    help='gradient clipping')
parser.add_argument('--epoch', '-ep', action='store', type=int, default=1,
                    help='# of training epochs')
parser.add_argument('--cur_epoch', action='store', type=int, default=1,
                    help='current epochs')
parser.add_argument('--min_step', '-minstep', action='store', type=int, default=1500,
                    help='# of step for joint training')


parser.add_argument('--hidden_size', '-hs', action='store', type=int, default=768,
                    help='# of training epochs')
parser.add_argument('--dp', '-dp', action='store', type=float, default=0.1,
                    help='# of training epochs')
parser.add_argument('--ddp', '-ddp', action='store', type=float, default=0.5,
                    help='domain classifeir dropout probability')
parser.add_argument('--thre', '-th', action='store', type=float, default=0.5,
                    help='threshold for multi-label classification')

parser.add_argument('--pre_every', '-pre_every',
                    action='store', type=int, default=100)
parser.add_argument('--fewshot_every', '-fs_every',
                    action='store', type=int, default=10)
parser.add_argument('--warmup', '-warm',
                    action='store', type=int, default=250)

parser.add_argument('--few_shot', '-fs', action='store_true', default=False)
parser.add_argument('--num_few_shot', '-nfs', action='store',
                    type=int, default=-1, help='# of few shot examples')
parser.add_argument('--use_src_f1', '-use_src', action='store_true', default=False)

parser.add_argument('--skip_pre', '-spre', action='store_true', default=False)
parser.add_argument('--skip_training', '-strain',
                    action='store_true', default=False)

parser.add_argument('--n_gpu', action='store', type=int,
                    default=0, help='logging purpose: n_gpu')
parser.add_argument('--best_zero_src_f1', action='store', type=float,
                    default=0.0, help='logging purpose: best_zero_src_f1')
parser.add_argument('--best_zero_tgt_f1', action='store', type=float,
                    default=0.0, help='logging purpose: best_zero_tgt_f1')
parser.add_argument('--best_f1', action='store', type=float,
                    default=0.0, help='logging purpose: best_f1')
parser.add_argument('--best_src_f1', action='store', type=float,
                    default=0.0, help='logging purpose: best_src_f1')
parser.add_argument('--best_tgt_f1', action='store', type=float,
                    default=0.0, help='logging purpose: best_tgt_f1')
parser.add_argument('--best_tgt_epoch', action='store', type=int,
                    default=0, help='logging purpose: best_tgt_epoch')
args = parser.parse_args()

assert not args.target in args.source, f"{args.source} must not include {args.target}"

if args.bert_path_model == "blm":
    args.bert_path = PATH_BERT_BLM
elif args.bert_path_model == "blmonly":
    args.bert_path = PATH_BERT_BLM_ONLY
elif args.bert_path_model == "hur":
    args.bert_path = PATH_BERT_HUR
elif args.bert_path_model == "huronly":
    args.bert_path = PATH_BERT_HUR_ONLY

if args.save:
    file_name = ""
    if args.append_timestamp:
        timestamp = datetime.now().strftime('%m-%d-%H-%M')
        file_name += timestamp + "_"
        args.timestamp = timestamp
    file_name += args.target_emotion
    if args.suffix != "":
        file_name += "_" + args.suffix
    save_path = os.path.join(args.model_path, file_name)
    try:
        os.mkdir(save_path)
    except:
        time.sleep(60)
        timestamp = datetime.now().strftime('%m-%d-%H-%M')
        save_path = os.path.join(args.model_path, timestamp+args.suffix)
        os.mkdir(save_path)
    
    args.save_path = save_path