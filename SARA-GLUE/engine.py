
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

def eval_func(preds, labels, task=config.CONFIG.task):
    if task in ['sst2', 'rte', 'mrpc', 'cola', 'qnli', 'qqp', 'mnli']:
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
        pearson_corr = stats.pearsonr(labels_flat, preds_flat)
        spearman_corr = stats.spearmanr(labels_flat, preds_flat)
        return {'pearson': pearson_corr, 'spearman': spearman_corr}

    else:
        raise ValueError("Task not recognized. Supported tasks are: 'sst2', 'rte', 'mrpc', 'cola', 'stsb'.")

   



def evaluate(model, val_dataloader):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in val_dataloader:


        inputs = {'input_ids':      batch['input_ids'].to(config.CONFIG.device),
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

    loss_val_avg = loss_val_total/len(val_dataloader)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals    




def train(model, optimizer, scheduler, train_dataloader, val_dataloader):

    epochs = config.CONFIG.epochs
    model.to(config.CONFIG.device)

    val_acc_list = []

    for epoch in tqdm(range(1, epochs+1)):
      
      model.train()

      loss_train_total = 0

      progress_bar = tqdm(train_dataloader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=True)

      for batch in progress_bar:

        optimizer.zero_grad()

        inputs = {'input_ids':      batch['input_ids'].to(config.CONFIG.device),
                  'attention_mask': batch['attention_mask'].to(config.CONFIG.device),
                  'labels':         batch['labels'].to(config.CONFIG.device),
                }

        output = model(**inputs)

        loss = output["loss"]
        loss_train_total += loss.item()
        loss.backward()


        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})


      tqdm.write(f'\nEpoch {epoch}')
      loss_train_avg = loss_train_total/len(train_dataloader)
      tqdm.write(f'Training loss: {loss_train_avg}')


      val_loss, predictions, true_vals = evaluate(model, val_dataloader)
      val_f1 = eval_func(predictions, true_vals)
      val_acc_list.append(val_f1)
      tqdm.write(f'Validation loss: {val_loss}')
      tqdm.write(f'Accuracy : {val_f1}')

    print("\nMaximum ACC:", max(val_acc_list))  
