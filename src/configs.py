PATH_ROOT="/projects/tir1/users/chanyoun/emotion-classification"

# Data Path
PATH_PROCESSED = PATH_ROOT+"/data/processed-emotions/"
# PATH_HURRICANE = PATH_DATA+"/data/hurricane/datasets_raw/"
# PATH_HURRICANE_PROCESSED = PATH_DATA+"/data/hurricane/datasets_binary/"
# PATH_GOEMOTIONS = PATH_DATA+"/data/goemotions/data/"
AVAILABLE_DATA = ["blm","go","hurricane","sri"]

# Emotion-related info (mapping & emotions)
PATH_MAPPINGS = PATH_ROOT+"/data/emo-mapping/"
EMO_EKMAN = ["disgust", "fear", "anger", "sadness", "surprise", "joy"]
EMO_PLUTCHIK = ["disapproval", "aggressiveness", "optimism",
                "remorse", "love", "awe", "contempt", "submission"]

# Model-related Save Paths
PATH_MODELS = PATH_ROOT+"/saved_models/"
PATH_SUMMARY = PATH_ROOT+"/model_summary/"

# save paths for the generated labels
PATH_SAVE_DIR = PATH_ROOT+"/res/"

# Bert-related configuration/paths
BERT_MAX_LENGTH = 60
BERT_DIR = "./transformer-models/"
PATH_BERT_BLM = BERT_DIR+"bert_blm_from-pre-trained/best_model"
PATH_BERT_BLM_ONLY = BERT_DIR+"bert_blm/best_model"
PATH_BERT_HUR = BERT_DIR+"bert_hurricane_ext_from-pre-trained/best_model"
PATH_BERT_HUR_ONLY = BERT_DIR+"bert_hurricane_ext/best_model"
