
import config
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

import config
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

class GLUEDataset(Dataset):
    def __init__(self, dataset_name=config.CONFIG.task, split="train", tokenizer_name=config.CONFIG.model_name, max_len=config.CONFIG.max_len, k=None, seed=42):
        self.dataset_name = dataset_name
        self.split = split
        self.max_len = max_len
        self.k = k
        self.seed = seed
        
        if self.dataset_name == "sst2":
            self.dataset = load_dataset(dataset_name)[split].to_pandas()
        else:
            self.dataset = load_dataset('glue', dataset_name)[split].to_pandas()
        
        # Set the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Sample k random indices if k is specified
        if k is not None and split == "train":
            random.seed(seed)
            sampled_indices = random.sample(range(len(self.dataset)), k)
            self.dataset = self.dataset.iloc[sampled_indices].reset_index(drop=True)

        # Map dataset fields to class attributes based on task type
        if dataset_name in ['sst2', 'cola']:
            self.text = self.dataset['sentence'].values
            self.labels = self.dataset['label'].values
        elif dataset_name in ['mrpc', 'qqp', 'stsb', 'rte']:
            self.sentence1 = self.dataset['sentence1'].values
            self.sentence2 = self.dataset['sentence2'].values
            self.labels = self.dataset['label'].values
        elif dataset_name == 'mnli':
            self.premises = self.dataset['premise'].values
            self.hypotheses = self.dataset['hypothesis'].values
            self.labels = self.dataset['label'].values
        elif dataset_name == 'qnli':
            self.questions = self.dataset['question'].values
            self.sentences = self.dataset['sentence'].values
            self.labels = self.dataset['label'].values

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.dataset_name in ['sst2', 'cola']:
            text = self.text[index]
            inputs = self.tokenizer.encode_plus(
                text,
                None,
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_token_type_ids=True
            )
        elif self.dataset_name in ['mrpc', 'qqp', 'stsb', 'rte']:
            sentence1 = self.sentence1[index]
            sentence2 = self.sentence2[index]
            inputs = self.tokenizer.encode_plus(
                sentence1,
                sentence2,
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_token_type_ids=True
            )
        elif self.dataset_name == 'mnli':
            premise = self.premises[index]
            hypothesis = self.hypotheses[index]
            inputs = self.tokenizer.encode_plus(
                premise,
                hypothesis,
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_token_type_ids=True
            )
        elif self.dataset_name == 'qnli':
            question = self.questions[index]
            sentence = self.sentences[index]
            inputs = self.tokenizer.encode_plus(
                question,
                sentence,
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_token_type_ids=True
            )

        # Convert inputs to tensors
        inputs['input_ids'] = torch.tensor(inputs['input_ids'], dtype=torch.long)
        inputs['attention_mask'] = torch.tensor(inputs['attention_mask'], dtype=torch.long)

        if 'token_type_ids' in inputs:
            inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

        label = torch.tensor(self.labels[index], dtype=torch.long if self.dataset_name != 'stsb' else torch.float)

        result = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "labels": label
        }

        if 'token_type_ids' in inputs:
            result["token_type_ids"] = inputs['token_type_ids']

        return result
