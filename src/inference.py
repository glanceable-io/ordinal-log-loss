import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import List, Tuple
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
import csv
home_path = Path.home()
import torch
import random
from datasets import load_dataset
from scipy.stats import entropy
from scipy.special import softmax
from collections import Counter



def evaluate_model(labels: List[int], preds: List[int]) -> dict:
    """
    Evaluate a train supervised model
    Args:
        labels: list of labels
        preds: list of predictions

    Returns:

    """
    precision, recall, f1, _ = precision_recall_fscore_support(labels,
                                                               preds,
                                                               average='weighted')
    acc_labels = accuracy_score(labels, preds)
    cm_labels = confusion_matrix(y_true=labels, y_pred=preds)

    distances = [dist_matrix[preds[k]][labels[k]] for k in range(len(preds))]
    cnt = Counter(distances)
    
    dico_logs_["labels-accuracy"] = round(acc_labels, 4)
    dico_logs_["labels-precision"] = round(precision, 4)
    dico_logs_["labels-recall"] = round(recall, 4)
    dico_logs_["labels-f1_score"] = round(f1, 4)
    dico_logs_["labels-confusion_matrix"] = cm_labels

    for k in np.sort(list(cnt.keys())):
        dico_logs_[f"distance_{k}"] = round(cnt[k]/len(preds),4)

    return dico_logs_

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding='max_length', max_length=max_len)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding='max_length', max_length=max_len)


def preprocess_function_unbatched(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], max_length=max_len)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding='max_length', max_length=max_len)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    datasets = {"mnli": {"num_classes": 3, "task": ("premise", "hypothesis"), "tok_len": 128,
                        "int2label": ["entailment","neutral","contradiction"],
                        "dist": [[0,1,2],[1,0,1],[2,1,0]]},
                "sst5": {"num_classes": 5, "task": ("sentence", None), "tok_len": 128,
                        "int2label": [1,2,3,4,5],
                        "dist": [[0,1,2,3,4],[1,0,1,2,3],[2,1,0,1,2],[3,2,1,0,1],[4,3,2,1,0]]},
                "emotion": {"num_classes": 6, "task": ("text", None), "tok_len": 128, 
                            "int2label": ["sadness", "joy", "love", "anger", "fear", "surprise"],
                            "dist": [[0,2,2,1,1,2],[2,0,1,2,2,1],[2,1,0,2,2,1],[1,2,2,0,1,2],[1,2,2,1,0,2],[2,1,1,2,2,0]]}
                }


    # Loading the models
    for dataset_file in ["sst5", "emotion","mnli"]:#list(datasets.keys()):
        num_classes = datasets[dataset_file]["num_classes"]
        max_len = datasets[dataset_file]["tok_len"]
        sentence1_key, sentence2_key = datasets[dataset_file]["task"]
        dist_matrix = datasets[dataset_file]["dist"]
        data_path = f"{Path.home()}/glanceable-research/data/datasets/loss_research/{dataset_file}"
        dataset = load_dataset('csv', data_files={'test':f"{data_path}/{dataset_file}_test.csv"})
        
        models_path = f"{Path.home()}/glanceable-research/src/research/ordinal_loss_classification/output_models/{dataset_file}/saved_models"
        saved_models = np.sort(os.listdir(models_path))
        
        model_checkpoint = "TO COMPLETE"
                
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(device)
        

        encoded_dataset = dataset.map(preprocess_function_unbatched, batched=False)
            
        batch_size = 1
        predictions_test = []
        print("predicting")
        input_ids =  torch.tensor(encoded_dataset["test"]["input_ids"]).to(device)
        attention_mask = torch.tensor(encoded_dataset["test"]["attention_mask"]).to(device)
        for k in tqdm(range(0,len(encoded_dataset["test"]),batch_size)):
            input_ids_batch =  input_ids[k:k+batch_size]
            attention_mask_batch = attention_mask[k:k+batch_size]
            preds = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
            predictions_test.extend(preds.logits.argmax(dim=1).tolist())
            
            
        dico_logs_ = {}
        evaluate_model(labels=encoded_dataset["test"]["label"], preds=predictions_test)
        
        print(dico_logs_)