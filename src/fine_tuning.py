from scipy.special import expit
import json
from collections import Counter
import random
from datasets import load_dataset
import csv
from tqdm import tqdm
import numpy as np
from src.model_coral import CoralModel
from src.loss_functions import *
import os
import sys
import torch
from typing import Tuple, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from pathlib import Path
sys.path.append(os.getcwd())



def compute_metrics_coral(pred):
    labels = pred.label_ids
    preds = (np.column_stack((np.zeros((pred.predictions.shape[0], 1)), expit(
        pred.predictions))) > 0.5).sum(axis=1)
    # Compute metrics for labels
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted')
    acc_labels = accuracy_score(labels, preds)
    cm_labels = confusion_matrix(y_true=labels, y_pred=preds)

    distances = [dist_matrix[preds[k]][labels[k]] for k in range(len(preds))]
    cnt = Counter(distances)

    dico_logs_["labels-accuracy"] = round(acc_labels, 4)
    dico_logs_["labels-precision"] = round(precision, 4)
    dico_logs_["labels-recall"] = round(recall, 4)
    dico_logs_["labels-f1_score"] = round(f1, 4)
    dico_logs_["labels-confusion_matrix"] = cm_labels

    for k in np.sort(list(cnt.keys()))[1:]:
        dico_logs_[f"distance_{k}"] = round(cnt[k]/len(preds), 4)

    acc = acc_labels
    for k in np.sort(list(cnt.keys()))[1:-1]:
        acc += cnt[k]/len(preds)
        dico_logs_[f"off-by-{k}-accuracy"] = round(acc, 4)

    return dico_logs_


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Compute metrics for labels
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted')
    acc_labels = accuracy_score(labels, preds)
    cm_labels = confusion_matrix(y_true=labels, y_pred=preds)

    distances = [dist_matrix[preds[k]][labels[k]] for k in range(len(preds))]
    cnt = Counter(distances)

    dico_logs_["labels-accuracy"] = round(acc_labels, 4)
    dico_logs_["labels-precision"] = round(precision, 4)
    dico_logs_["labels-recall"] = round(recall, 4)
    dico_logs_["labels-f1_score"] = round(f1, 4)
    dico_logs_["labels-confusion_matrix"] = cm_labels

    for k in np.sort(list(cnt.keys()))[1:]:
        dico_logs_[f"distance_{k}"] = round(cnt[k]/len(preds), 4)

    acc = acc_labels
    for k in np.sort(list(cnt.keys()))[1:-1]:
        acc += cnt[k]/len(preds)
        dico_logs_[f"off-by-{k}-accuracy"] = round(acc, 4)

    return dico_logs_


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding='max_length', max_length=max_len)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding='max_length', max_length=max_len)


losses_dict = {"CE": Trainer,
               "OLL1": OLL1Trainer,
               "OLL15": OLL15Trainer,
               "OLL2": OLL2Trainer,
               "WKL": WKLTrainer,
               "SOFT2": SOFT2Trainer,
               "SOFT3": SOFT3Trainer,
               "SOFT4": SOFT4Trainer,
               "EMD": EMDTrainer,
               "CORAL": Trainer}


if __name__ == '__main__':

    device = torch.device('cuda:0')
    root_path = f"{Path.home()}/ordinal_loss_research"

    with open(f"{root_path}/src/datasets.json", "r") as f:
        datasets = json.load(f)

    learning_rates = [1e-4, 7.5e-5, 5e-5, 2.5e-5, 1e-5]
    
    losses = ["CE","OLL1","OLL15","OLL2","WKL","SOFT2","SOFT3","SOFT4","EMD","CORAL"]
    datasets_list = ["sst5", "amazon_reviews", "yelp", "snli"]


    # Loading the model and tokenizer
    model_checkpoint = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    for data_file in datasets_list:
        for loss_type in losses:
            for learning_rate_ in learning_rates:
                num_classes = datasets[data_file]["num_classes"]
                max_len = datasets[data_file]["tok_len"]
                sentence1_key, sentence2_key = datasets[data_file]["task"]
                dist_matrix = datasets[data_file]["dist"]
                directory_path = datasets[data_file]["path"]

                # Load the dataset as a DatasetDict
                data_path = f"{directory_path}/{data_file}"
                dataset = load_dataset('csv', data_files={'train': f"{data_path}/{data_file}_train.csv",
                                       "validation": f"{data_path}/{data_file}_validation.csv", 'test': f"{data_path}/{data_file}_test.csv"})

                # Tokenize dataset
                encoded_dataset = dataset.map(
                    preprocess_function, batched=True)

                dico_logs_ = {}

                # Training parameters
                weight_decay_ = 0.01
                train_batch_size_ = 1024
                valid_batch_size_ = 1024

                # Smaller datasets require more epochs for the model to converge
                epochs_ = int(20000000/len(dataset['train']))
                # If the model does not perform better for more than 5% of the total epochs in a row, then the training stops
                stopping_rate = int(0.05*epochs_)

                model_name = model_checkpoint+"-"+loss_type
                for k in tqdm(range(1, 6)):
                    random.seed(k)
                    model_name = "-".join([model_checkpoint,
                                          data_file, loss_type, str(k)])

                    # We check that the model has not already been trained
                    if Path(f"{root_path}/src/outputs_training/output_models/{data_file}/saved_models/{model_name}_{epochs_}_ep_{learning_rate_}_lr_{train_batch_size_}_batch").is_dir():
                        continue

                    # load model and initialize parameters
                    if loss_type == "CORAL":
                        model = CoralModel.from_pretrained(
                            model_checkpoint, num_labels=num_classes).to(device)
                    else:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_checkpoint, num_labels=num_classes).to(device)
                    model.dist_matrix = dist_matrix

                    print(
                        f'Epochs: {epochs_} | Learning rate: {learning_rate_}')
                    dico_logs_["model_name"] = model_name
                    dico_logs_["epochs"] = epochs_
                    dico_logs_["learning_rate"] = learning_rate_
                    dico_logs_["train_batch_size"] = train_batch_size_

                    training_args = TrainingArguments(
                        # output directory
                        output_dir=f"{root_path}/src/outputs_training/output_models/{data_file}/training/{model_name}/",
                        fp16=True,
                        num_train_epochs=epochs_,
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        load_best_model_at_end=True,
                        logging_steps=50,
                        save_total_limit=1,
                        # batch size per device during training
                        per_device_train_batch_size=train_batch_size_,
                        per_device_eval_batch_size=valid_batch_size_,   # batch size for evaluation
                        learning_rate=learning_rate_,
                        # directory for storing logs
                        logging_dir=f"{root_path}/src/outputs_training/output_models/{data_file}/logs/{model_name}_{epochs_}_ep_{learning_rate_}_lr_{train_batch_size_}_batch",
                    )

                    loss_function = losses_dict[loss_type]

                    if loss_type == "CORAL":
                        eval_function = compute_metrics_coral
                    else:
                        eval_function = compute_metrics

                    trainer = loss_function(
                        # the instantiated 🤗 Transformers model to be trained
                        model=model,
                        args=training_args,             # training arguments, defined above
                        compute_metrics=eval_function,
                        # training dataset
                        train_dataset=encoded_dataset["train"],
                        eval_dataset=encoded_dataset["validation"],
                        # evaluation dataset
                        callbacks=[EarlyStoppingCallback(
                            early_stopping_patience=stopping_rate)]
                    )

                    print('--- TRAINING ---')
                    trainer.train()
                    print('--- EVALUATION ---')
                    trainer.evaluate()

                    # The model is saved
                    trainer.save_model(
                        f"{root_path}/src/outputs_training/output_models/{data_file}/saved_models/{model_name}_{epochs_}_ep_{learning_rate_}_lr_{train_batch_size_}_batch")

                    # Evaluation metrics are saved to a csv file
                    output_path_metrics = f'{root_path}/src/outputs_training/output_metrics/fine_tuning_metrics.csv'
                    print(list(dico_logs_.keys()))
                    values = [dico_logs_[key] for key in dico_logs_.keys(
                    ) if key != "labels-confusion_matrix"]
                    with open(output_path_metrics, "a+") as f:
                        writer = csv.writer(f)
                        writer.writerow(values)
