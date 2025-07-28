from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class CONFIG:
    output_dir = "SHA-DIAG"
    task = "cola"
    seed = 46
    max_len = 256
    train_batch = 32
    valid_batch = 32
    epochs = 40
    learning_rate = 6e-3
    classifier_learning_rate = 1e-4
    warmup_ratio = 0.06
    r = 8
    c = 4
    alpha = 8
    dropout = 0.1
    k = None
    model_name = "FacebookAI/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/home/m_azimi/.cache/huggingface")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
