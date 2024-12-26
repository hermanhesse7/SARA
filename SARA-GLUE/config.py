
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class CONFIG:
    output_dir = "SHA-DIAG"
    task = "mnli"
    seed = 44
    max_len = 128
    train_batch = 64
    valid_batch = 64
    epochs = 30
    learning_rate = 4e-3
    classifier_learning_rate = 2e-3
    warmup_ratio = 0.06
    r = 8
    c = 1
    alpha = 64
    dropout = 0.1
    k = 20000
    model_name = "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
