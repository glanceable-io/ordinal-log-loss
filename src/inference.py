import os
from typing_extensions import ParamSpecArgs
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_absolute_error, mean_squared_error
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
from scipy.stats import entropy, kendalltau
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
    



    for k in range(1,n_distances):
        dico_logs_[f"distance_{k}"] = round(cnt[k]/len(preds),4)

    acc = acc_labels
    for k in range(1,n_distances-1):
        acc += cnt[k]/len(preds)
        dico_logs_[f"off-by-{k}-accuracy"] = round(acc,4)

    dico_logs_["mae"] = mean_absolute_error(y_true=labels, y_pred=preds)
    dico_logs_["mse"] = mean_squared_error(y_true=labels, y_pred=preds)
    dico_logs_["kendalltau"], _ = kendalltau(labels,preds)

    repartitions = Counter(preds)
    for k in range(len(repartitions)):
        dico_logs_[f"distrib-{k}"] = repartitions[k]

    return dico_logs_

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding='max_length', max_length=max_len)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding='max_length', max_length=max_len)


def preprocess_function_unbatched(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], max_length=max_len)
    return tokenizer(examples[sentence1_key], examples[sentence2_key],max_length=max_len)


def get_distributions(distributions: list, labels: list, correct: bool):
    """Return distributions either if the predictions are correct or incorrect

    Args:
        correct (bool): [description]
    """
    final_distributions = []

    for dist, lab in zip(distributions, labels):
        if int(np.argmax(dist)) == lab and correct:
            final_distributions.append(dist)
        if int(np.argmax(dist)) != lab and not correct:
            final_distributions.append(dist)

    return final_distributions


if __name__ == '__main__':
    device = torch.device('cuda:0')
    output_path_metrics = f'{Path.home()}/ordinal_loss_research/src/outputs_training/output_metrics/metrics_test_set_v4.csv'
    datasets = {"snli": {"num_classes": 3, "task": ("premise", "hypothesis"), "tok_len": 128,
                        "int2label": ["entailment","neutral","contradiction"],
                        "dist": [[0,1,2],[1,0,1],[2,1,0]],
                        "n_distances": 3},
                "sst5": {"num_classes": 5, "task": ("sentence", None), "tok_len": 128,
                        "int2label": [1,2,3,4,5],
                        "dist": [[0,1,2,3,4],[1,0,1,2,3],[2,1,0,1,2],[3,2,1,0,1],[4,3,2,1,0]],
                        "n_distances": 5},
                "emotion": {"num_classes": 6, "task": ("text", None), "tok_len": 128, 
                            "int2label": ["sadness", "joy", "love", "anger", "fear", "surprise"],
                            "dist": [[0,2,2,1,1,2],[2,0,1,2,2,1],[2,1,0,2,2,1],[1,2,2,0,1,2],[1,2,2,1,0,2],[2,1,1,2,2,0]],
                        "n_distances": 3},
                "amazon_reviews": {"num_classes": 5, "task": ("text", None), "tok_len": 128,
                        "int2label": [1,2,3,4,5],
                        "dist": [[0,1,2,3,4],[1,0,1,2,3],[2,1,0,1,2],[3,2,1,0,1],[4,3,2,1,0]],
                        "n_distances": 5},
                "yelp": {"num_classes": 5, "task": ("text", None), "tok_len": 128,
                        "int2label": [1,2,3,4,5],
                        "dist": [[0,1,2,3,4],[1,0,1,2,3],[2,1,0,1,2],[3,2,1,0,1],[4,3,2,1,0]],
                        "n_distances": 5}
                }
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

    # Loading the models
    for dataset_file in ["snli","amazon_reviews","sst5","yelp"]:#list(datasets.keys()):
        num_classes = datasets[dataset_file]["num_classes"]
        n_distances = datasets[dataset_file]["n_distances"]
        max_len = datasets[dataset_file]["tok_len"]
        sentence1_key, sentence2_key = datasets[dataset_file]["task"]
        dist_matrix = datasets[dataset_file]["dist"]

        data_path = f"/home/francois/glanceable-research/data/datasets/loss_research/{dataset_file}"
        dataset = load_dataset('csv', data_files={'test':f"{data_path}/{dataset_file}_test.csv"})
        
        model_dir = "google/"
        models_path = f"/home/francois/ordinal_loss_research/src/outputs_training/output_models/{dataset_file}/saved_models/{model_dir}"
        saved_models = np.sort(os.listdir(models_path))
        dictpath = {}
        offset = 28
        for path in saved_models:
            if "CE" in path : 
                loss_len = 2
            elif "OLL15" in path  or "SOFT2" in path or "SOFT3" in path or "SOFT4" in path or "nOLL2" in path: 
                loss_len = 5
            elif "SOFT10" in path : 
                loss_len = 6
            elif "WKL" in path : 
                loss_len = 3
            else  : 
                loss_len = 4
            n = len(dataset_file)+loss_len+offset
            corrected_path = path[:n] + path[n+2:] + path[n:n+2]

            dictpath[corrected_path] = path

        

        if os.path.isfile(output_path_metrics):
            dt = pd.read_csv(output_path_metrics, header=None, sep='\n')
            dt = dt[0].str.split(',', expand=True)
        else : 
            dt = 5*['']

        for path in np.sort(list(dictpath.keys())):
            trained_model = dictpath[path]

            if trained_model in list(dt[3]): 
                continue

            if "-CE-" in  trained_model:
                loss_func = "CE"
                
                
            elif "-OLL2-" in  trained_model:
                loss_func = "OLL2"

            elif "-nOLL2-" in  trained_model:
                loss_func = "nOLL2"
                
                
            elif "-OLL1-" in  trained_model:
                loss_func = "OLL1"
                
                
            elif "-OLL15-" in  trained_model:
                loss_func = "OLL1.5"
                
                
            elif "-WKL-" in  trained_model:
                loss_func = "WKL"

            elif "-SOFT10-" in  trained_model:
                loss_func = "SOFT10"
                
            elif "-SOFT2-" in  trained_model:
                loss_func = "SOFT2"
            
            elif "-SOFT3-" in  trained_model:
                loss_func = "SOFT3"

            elif "-SOFT4-" in  trained_model:
                loss_func = "SOFT4"

            if "bert-tiny" in trained_model or "bert_uncased_L-2_H-128_A-2" in trained_model: 
                pre_trained_model = f"{model_dir}bert-tiny"
                
            #tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
            model = AutoModelForSequenceClassification.from_pretrained(f"{models_path}/{trained_model}").to(device)
            

            encoded_dataset = dataset.map(preprocess_function_unbatched, batched=False)
                
            batch_size = 1
            predictions_test, distributions = [], []
            print("predicting")
            #input_ids =  torch.tensor(encoded_dataset["test"]["input_ids"]).to(device)
            #attention_mask = torch.tensor(encoded_dataset["test"]["attention_mask"]).to(device)
            for k in tqdm(range(0,len(encoded_dataset["test"]),batch_size)):
                inputs = {"input_ids": encoded_dataset["test"][k:k+batch_size]["input_ids"],"attention_mask": encoded_dataset["test"][k:k+batch_size]["attention_mask"],"token_type_ids": encoded_dataset["test"][k:k+batch_size]["token_type_ids"]}
                preds = model(torch.tensor(inputs["input_ids"]).to(device), attention_mask = torch.tensor(inputs["attention_mask"]).to(device), token_type_ids = torch.tensor(inputs["token_type_ids"]).to(device))
                #preds = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
                distributions.extend(softmax(preds.logits.cpu().detach().numpy(), axis=1).tolist())
                predictions_test.extend(preds.logits.argmax(dim=1).tolist())
                
            # Get the distributions with correct predictions
            #correct_distributions = get_distributions(distributions=distributions, labels=encoded_dataset["test"]["label"], correct=True)
            # Get the distributions with incorrect predictions
            #incorrect_distributions = get_distributions(distributions=distributions, labels=encoded_dataset["test"]["label"], correct=False)
            #a = 1
            
            dico_logs_ = {}
            evaluate_model(labels=encoded_dataset["test"]["label"], preds=predictions_test)
            
            new_row = [dataset_file,loss_func,pre_trained_model,trained_model] + [dico_logs_[k] for k in dico_logs_.keys() if k != "labels-confusion_matrix"]
        
            
            with open(output_path_metrics, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(new_row)
            