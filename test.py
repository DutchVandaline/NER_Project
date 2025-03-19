import os
import json
import re
import torch
import numpy as np
import pandas as pd
import string
from collections import Counter
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, BertForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns




os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def extract_entities(labels, tokens, tag):
    entities = []
    current_entity_tokens = []
    for i, (label, token) in enumerate(zip(labels, tokens)):
        token_clean = token[2:] if token.startswith("##") else token
        if label == f"B-{tag}":
            if current_entity_tokens:
                entities.append("".join(current_entity_tokens))
            current_entity_tokens = [token_clean]
        elif label == f"I-{tag}" and current_entity_tokens:
            # 서브워드인 경우 앞 토큰과 붙여서
            if token.startswith("##"):
                current_entity_tokens.append(token_clean)
            else:
                if token in string.punctuation:
                    current_entity_tokens.append(token_clean)
                else:
                    if current_entity_tokens and current_entity_tokens[-1][-1] in string.punctuation:
                        current_entity_tokens.append(token_clean)
                    else:
                        current_entity_tokens.append(" " + token_clean)
        else:
            if current_entity_tokens:
                entities.append("".join(current_entity_tokens))
                current_entity_tokens = []
    if current_entity_tokens:
        entities.append("".join(current_entity_tokens))
    return " ".join(entities)


