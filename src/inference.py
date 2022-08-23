from src.model_coral import CoralModel
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_absolute_error, mean_squared_error
from typing import List
import numpy as np
from tqdm import tqdm
import csv
import torch
from datasets import load_dataset
from scipy.stats import kendalltau
from scipy.special import softmax, expit
from collections import Counter
import json
import sys
sys.path.append(os.getcwd())


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

    for k in range(1, n_distances):
        dico_logs_[f"distance_{k}"] = round(cnt[k]/len(preds), 4)

    acc = acc_labels
    for k in range(1, n_distances-1):
        acc += cnt[k]/len(preds)
        dico_logs_[f"off-by-{k}-accuracy"] = round(acc, 4)

    dico_logs_["mae"] = mean_absolute_error(y_true=labels, y_pred=preds)
    dico_logs_["mse"] = mean_squared_error(y_true=labels, y_pred=preds)
    dico_logs_["kendalltau"], _ = kendalltau(labels, preds)

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
    return tokenizer(examples[sentence1_key], examples[sentence2_key], max_length=max_len)


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
    root_path = f"{Path.home()}/ordinal_loss_research"

    with open(f"{root_path}/src/datasets.json", "r") as f:
        datasets = json.load(f)

    output_path_metrics = f'{root_path}/src/outputs_training/output_metrics/metrics_test_set.csv'

    model_checkpoint = "google/bert_uncased_L-2_H-128_A-2"
    model_dir = "google/"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    for dataset_file in ["amazon_reviews", "snli", "sst5", "yelp"]:
        num_classes = datasets[dataset_file]["num_classes"]
        n_distances = datasets[dataset_file]["n_distances"]
        max_len = datasets[dataset_file]["tok_len"]
        sentence1_key, sentence2_key = datasets[dataset_file]["task"]
        dist_matrix = datasets[dataset_file]["dist"]
        directory_path = datasets[dataset_file]["path"]

        data_path = f"{directory_path}/{dataset_file}"
        dataset = load_dataset(
            'csv', data_files={'test': f"{data_path}/{dataset_file}_test.csv"})

        models_path = f"{root_path}/src/outputs_training/output_models/{dataset_file}/saved_models/{model_dir}"
        saved_models = np.sort(os.listdir(models_path))
        dictpath = {}
        offset = 28
        losses = ["CE", "OLL15", "OLL1", "OLL2", "WKL",
                  "SOFT2", "SOFT3", "SOFT4", "EMD", "CORAL"]
        for path in saved_models:
            for loss in losses:
                if loss in path:
                    loss_len = len(loss)
                    loss_name = loss
                    continue

            # we correct each path to have results sorted by dataset, loss, learning rate
            n = len(dataset_file)+loss_len+offset
            corrected_path = path[:n] + path[n+2:] + path[n:n+2]

            dictpath[corrected_path] = {"path": path, "loss": loss_name}

        if os.path.isfile(output_path_metrics):
            dt = pd.read_csv(output_path_metrics, header=None, sep='\n')
            dt = dt[0].str.split(',', expand=True)
        else:
            dt = 5*['']

        for path in np.sort(list(dictpath.keys())):
            trained_model = dictpath[path]["path"]
            loss_func = dictpath[path]["loss"]

            # Check if evaluation has already been done for this
            if trained_model in list(dt[3]):
                continue

            if "bert-tiny" in trained_model or "bert_uncased_L-2_H-128_A-2" in trained_model:
                pre_trained_model = f"{model_dir}bert-tiny"

            if loss_func == "CORAL":
                model = CoralModel.from_pretrained(
                    f"{models_path}/{trained_model}").to(device)

            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    f"{models_path}/{trained_model}").to(device)

            encoded_dataset = dataset.map(
                preprocess_function_unbatched, batched=False)

            batch_size = 1
            predictions_test, distributions = [], []

            # Make predictions
            for k in tqdm(range(0, len(encoded_dataset["test"]), batch_size), "Predicting"):
                inputs = {"input_ids": encoded_dataset["test"][k:k+batch_size]["input_ids"], "attention_mask": encoded_dataset["test"]
                          [k:k+batch_size]["attention_mask"], "token_type_ids": encoded_dataset["test"][k:k+batch_size]["token_type_ids"]}
                preds = model(torch.tensor(inputs["input_ids"]).to(device), attention_mask=torch.tensor(
                    inputs["attention_mask"]).to(device), token_type_ids=torch.tensor(inputs["token_type_ids"]).to(device))
                if loss_func == "CORAL":
                    predictions_test.extend((np.column_stack((np.zeros((preds.logits.cpu().detach().numpy(
                    ).shape[0], 1)), expit(preds.logits.cpu().detach().numpy()))) > 0.5).sum(axis=1))
                else:
                    distributions.extend(
                        softmax(preds.logits.cpu().detach().numpy(), axis=1).tolist())
                    predictions_test.extend(
                        preds.logits.argmax(dim=1).tolist())

            # Evaluate model based on predictions
            dico_logs_ = {}
            evaluate_model(
                labels=encoded_dataset["test"]["label"], preds=predictions_test)

            new_row = [dataset_file, loss_func, pre_trained_model, trained_model] + \
                [dico_logs_[k]
                    for k in dico_logs_.keys() if k != "labels-confusion_matrix"]

            with open(output_path_metrics, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(new_row)
