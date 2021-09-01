import pandas as pd
import torch
from typing import Tuple, List
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import itertools
home_path = Path.home()
from tqdm import tqdm
import csv
from datasets import load_dataset
import random
from torch.autograd import Variable
from knockknock import slack_logs_sender
import requests
import json
from collections import Counter
from torch.utils import tensorboard





class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)




webhook_url = "https://hooks.slack.com/services/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"
#@slack_logs_sender(webhook_url=webhook_url, channel="glanceable-training",file_path = __file__)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Compute metrics for labels
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
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
        dico_logs_[f"distance_{k}"] = round(cnt[k]/len(preds),4)
    
    acc = acc_labels
    for k in np.sort(list(cnt.keys()))[1:-1]:
        acc += cnt[k]/len(preds)
        dico_logs_[f"off-by-{k}-accuracy"] = round(acc,4)

    return dico_logs_

def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=max_input_length, truncation=True)

    return model_inputs



class MSETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [int(labels[k].float().item()) for k in range(len(labels))]
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(np.array([k for k in range(num_classes)]).reshape(-1,1))
        true_probas = enc.transform(np.array(true_labels).reshape(-1,1)).toarray()
        true_probas_tensor = torch.tensor(true_probas,device='cuda:0', requires_grad=True)
        err = (probas - true_probas_tensor)**2
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class MSENSTrainer(Trainer): #No softmax
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        true_labels = [int(labels[k].float().item()) for k in range(len(labels))]
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(np.array([k for k in range(num_classes)]).reshape(-1,1))
        true_probas = enc.transform(np.array(true_labels).reshape(-1,1)).toarray()
        true_probas_tensor = torch.tensor(true_probas,device='cuda:0', requires_grad=True)
        err = (logits - true_probas_tensor)**2
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss


class OLL2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*abs(distances_tensor)**2
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class OLL4Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*abs(distances_tensor)**4
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss



def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding='max_length', max_length=max_len)
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

    learning_rates = [1e-5, 2.5e-5, 5e-5, 1e-4, 5e-4]
    #
    start_time = time.time()
    # Loading the data
    # We import the data as a DatasetDict
    for data_file in ["sst5","emotion","mnli"]:
        for loss_type in  ["CE","OLL2","OLL4"]:
            for learning_rate_ in learning_rates : 
            
                num_classes = datasets[data_file]["num_classes"]
                max_len = datasets[data_file]["tok_len"]
                sentence1_key, sentence2_key = datasets[data_file]["task"]
                dist_matrix = datasets[data_file]["dist"]
                data_path = f"{Path.home()}/glanceable-research/data/datasets/loss_research/{data_file}"
                dataset = load_dataset('csv', data_files={'train':f"{data_path}/{data_file}_train.csv", "validation":f"{data_path}/{data_file}_validation.csv",'test':f"{data_path}/{data_file}_test.csv"})
                
                # Loading the model and tokenizer
                model_checkpoint = "prajjwal1/bert-tiny"
                tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
                encoded_dataset = dataset.map(preprocess_function, batched=True)
                
                dico_logs_ = {}
                # Models parameters
                
                weight_decay_ = 0.01
                #warmup_steps_= 1000
                train_batch_size_ = 512
                valid_batch_size_ = 512
                epochs_ = int(10000000/len(dataset['train']))
                stopping_rate = int(0.05*epochs_)
                model_name = model_checkpoint+"-"+loss_type
                for k in tqdm([1,2,3,4,5]):
                    model_name = "-".join([model_checkpoint,data_file,loss_type,str(k)])
                    dump = {
                        "username": "Knock Knock",
                        "channel": "glanceable-training",
                        "icon_emoji": ":glanceable:",
                    }
                    #contents = [150*"*",150*"*",f"Starting Training number {k} of {data_file}",f"Pre-Trained Model : {model_checkpoint}", "Params :",f"Num Epochs: {epochs_}",f"Learning Rate: {learning_rate_}", f"Train Batch Size: {train_batch_size_}", f"Validation Batch Size: {valid_batch_size_}" ]
                    #dump['text'] = '\n'.join(contents)
                    #dump['icon_emoji'] = ':julom:'
                    #requests.post(webhook_url, json.dumps(dump))

                    #load model and initialize parameters
                    random.seed(k)
                    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels = num_classes).to(device)
                    
                    print(f'Epochs: {epochs_} | Learning rate: {learning_rate_}')
                    dico_logs_["model_name"] = model_name
                    dico_logs_["epochs"] = epochs_
                    dico_logs_["learning_rate"] = learning_rate_
                    #dico_logs_["weight_decay"] = weight_decay_
                    #dico_logs_["warmup_steps"] = warmup_steps_
                    dico_logs_["train_batch_size"] = train_batch_size_
                    
                    training_args = TrainingArguments(
                        output_dir=f"{Path.home()}/ordinal_loss_research/src/outputs_training/output_models/{data_file}/training/{model_name}/",          # output directory
                        fp16=True,
                        num_train_epochs=epochs_,
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        load_best_model_at_end=True,
                        metric_for_best_model="labels-accuracy",
                        logging_steps=50,
                        save_total_limit=1,
                        #eval_steps=500,              # total # of training epochs
                        per_device_train_batch_size=train_batch_size_,  # batch size per device during training
                        per_device_eval_batch_size=valid_batch_size_,   # batch size for evaluation
                        #warmup_steps=warmup_steps_,                # number of warmup steps for learning rate scheduler
                        #weight_decay=weight_decay_,               # strength of weight decay
                        learning_rate=learning_rate_,
                        logging_dir=f"{Path.home()}/ordinal_loss_research/src/outputs_training/output_models/{data_file}/logs/{model_name}_{epochs_}_ep_{learning_rate_}_lr_{train_batch_size_}_batch",            # directory for storing logs
                    )
                    if loss_type == "CE":
                        trainer = Trainer(
                            model=model,                         # the instantiated 🤗 Transformers model to be trained
                            args=training_args,             # training arguments, defined above
                            compute_metrics=compute_metrics,
                            train_dataset=encoded_dataset["train"],        # training dataset
                            eval_dataset=encoded_dataset["validation"],
                            callbacks=[EarlyStoppingCallback(early_stopping_patience=stopping_rate)]         # evaluation dataset
                        )

                    elif loss_type == "OLL2":
                        trainer = OLL2Trainer(
                            model=model,                         # the instantiated 🤗 Transformers model to be trained
                            args=training_args,             # training arguments, defined above
                            compute_metrics=compute_metrics,
                            train_dataset=encoded_dataset["train"],        # training dataset
                            eval_dataset=encoded_dataset["validation"],
                            callbacks=[EarlyStoppingCallback(early_stopping_patience=stopping_rate)]         # evaluation dataset
                        )
                    
                    elif loss_type == "OLL4":
                        trainer = OLL4Trainer(
                            model=model,                         # the instantiated 🤗 Transformers model to be trained
                            args=training_args,             # training arguments, defined above
                            compute_metrics=compute_metrics,
                            train_dataset=encoded_dataset["train"],        # training dataset
                            eval_dataset=encoded_dataset["validation"],
                            callbacks=[EarlyStoppingCallback(early_stopping_patience=stopping_rate)]         # evaluation dataset
                        )

                    print('--- TRAINING ---')
                    trainer.train()
                    print('--- EVALUATION ---')
                    trainer.evaluate()
                    
                    trainer.save_model(f"{Path.home()}/ordinal_loss_research/src/outputs_training/output_models/{data_file}/saved_models/{model_name}_{epochs_}_ep_{learning_rate_}_lr_{train_batch_size_}_batch")
                    print("Total process--- %s minutes ---" % ((time.time() - start_time)/60))
                    
                    dico_logs_['process_time_minutes'] = round((time.time() - start_time)/60, 4)
                    dump['text'] = ":tada:ENDED TRAINING:\n"+str(dico_logs_)
                    requests.post(webhook_url, json.dumps(dump))
                    output_path_metrics = f'{Path.home()}/ordinal_loss_research/src/outputs_training/output_metrics/fine_tuning_metrics.csv'
                    print(list(dico_logs_.keys()))
                    values = [dico_logs_[key] for key in dico_logs_.keys() if key!="labels-confusion_matrix"]
                    with open(output_path_metrics, "a+") as f:
                        writer = csv.writer(f)
                        writer.writerow(values)
