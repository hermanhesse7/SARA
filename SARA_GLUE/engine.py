# engine.py
import transformers
from transformers import AdamW
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy import stats
import config
import peft_module
import os

def eval_func(preds, labels, task=config.CONFIG.task):
    if task in ['sst2', 'rte', 'mrpc', 'cola', 'qnli', 'mnli', 'qqp']:
        preds_flat = np.argmax(preds, axis=1).flatten()
        
        if task == 'cola':
            # For CoLA, use MCC
            return matthews_corrcoef(labels, preds_flat)
        else:
            # For SST-2, RTE, and MRPC, use accuracy
            return accuracy_score(labels, preds_flat)
    
    elif task == 'stsb':
        # For STS-B, preds and labels are continuous values
        preds_flat = preds.flatten()
        labels_flat = labels.flatten()
        pearson_corr = stats.pearsonr(labels_flat, preds_flat)[0]
        spearman_corr = stats.spearmanr(labels_flat, preds_flat)[0]
        return {'pearson': pearson_corr, 'spearman': spearman_corr}

    else:
        raise ValueError("Task not recognized. Supported tasks are: 'sst2', 'rte', 'mrpc', 'cola', 'stsb'.")

def evaluate(model, val_dataloader):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in val_dataloader:
        inputs = {
            'input_ids':      batch['input_ids'].to(config.CONFIG.device),
            'attention_mask': batch['attention_mask'].to(config.CONFIG.device),
            'labels':         batch['labels'].to(config.CONFIG.device),
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs["loss"]
        logits = outputs["logits"]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(val_dataloader)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals    

def train(model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader=None):
    epochs = config.CONFIG.epochs
    model.to(config.CONFIG.device)

    val_metric_list = []
    best_val_metric = -np.inf  # Initialize to negative infinity
    best_epoch = -1
    best_model_path = "best_model_checkpoint.pt"

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        model.train()
        loss_train_total = 0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}', leave=False, disable=True)

        for batch in progress_bar:
            optimizer.zero_grad()

            inputs = {
                'input_ids':      batch['input_ids'].to(config.CONFIG.device),
                'attention_mask': batch['attention_mask'].to(config.CONFIG.device),
                'labels':         batch['labels'].to(config.CONFIG.device),
            }

            outputs = model(**inputs)
            loss = outputs["loss"]
            loss_train_total += loss.item()
            loss.backward()

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

        tqdm.write(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total / len(train_dataloader)
        tqdm.write(f'Training loss: {loss_train_avg:.4f}')

        val_loss, predictions, true_vals = evaluate(model, val_dataloader)
        val_f1 = eval_func(predictions, true_vals)
        val_metric_list.append(val_f1)

        _, predictions_test, true_tests = evaluate(model, test_dataloader)
        test_f1 = eval_func(predictions_test, true_tests)

        tqdm.write(f'Validation loss: {val_loss:.4f}')
        if isinstance(val_f1, dict):
            tqdm.write(f"Validation Metrics: {val_f1}")
            # For tasks like STS-B, you might want to use one of the metrics
            current_val_metric = val_f1.get('pearson', 0)  # Example: using Pearson correlation
        else:
            tqdm.write(f'Validation Metric: {val_f1:.4f}')
            current_val_metric = val_f1

        if isinstance(test_f1, dict):
            tqdm.write(f"Test Metrics: {test_f1}")
            # For tasks like STS-B, you might want to use one of the metrics
        else:
            tqdm.write(f'Test Metric: {test_f1:.4f}')


        # Check if current epoch has the best validation metric
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_epoch = epoch
            # Save the best model checkpoint
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f'Best model updated at epoch {epoch} with validation metric {best_val_metric:.4f}')

    print(f"\nTraining complete. Best validation metric: {best_val_metric:.4f} at epoch {best_epoch}")

    # Load the best model checkpoint
    model.load_state_dict(torch.load(best_model_path))

    # Optionally, evaluate on the test set if provided
    if test_dataloader is not None:
        test_loss, test_preds, test_true = evaluate(model, test_dataloader)
        test_metric = eval_func(test_preds, test_true)
        if isinstance(test_metric, dict):
            tqdm.write(f"Test Metrics: {test_metric}")
        else:
            tqdm.write(f'Test Metric: {test_metric:.4f}')

    return best_model_path  # Return the path to the best checkpoint