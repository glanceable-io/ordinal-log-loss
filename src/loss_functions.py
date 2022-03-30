import torch.nn.functional as F
import numpy as np
from transformers import Trainer
import torch
import sys
import os
sys.path.append(os.getcwd())
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from src.loss_functions import *




class OLL2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.module.num_labels
        dist_matrix = model.module.dist_matrix
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

class nOLL2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.module.num_labels
        dist_matrix = model.module.dist_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[dist_matrix[true_labels[j][i]][label_ids[j][i]]/np.sum([dist_matrix[n][label_ids[j][i]] for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*abs(distances_tensor)**2
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class WKLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.module.num_labels
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        y_pred = F.softmax(logits,dim=1)
        label_vec = torch.range(0,num_classes-1, dtype=torch.float)
        row_label_vec = torch.tensor(torch.reshape(label_vec, (1, num_classes)), requires_grad=True,device='cuda:0')
        col_label_vec = torch.tensor(torch.reshape(label_vec, (num_classes, 1)), requires_grad=True,device='cuda:0')
        col_mat = torch.tile(col_label_vec, (1, num_classes))
        row_mat = torch.tile(row_label_vec, (num_classes, 1))
        weight_mat = (col_mat - row_mat) ** 2
        y_true = torch.tensor(F.one_hot(labels, num_classes=num_classes), dtype=col_label_vec.dtype, requires_grad=True)
        batch_size = y_true.shape[0]
        cat_labels = torch.matmul(y_true, col_label_vec)
        cat_label_mat = torch.tensor(torch.tile(cat_labels, [1, num_classes]), requires_grad=True,device='cuda:0')
        row_label_mat = torch.tensor(torch.tile(row_label_vec, [batch_size, 1]), requires_grad=True,device='cuda:0')
        
        weight = (cat_label_mat - row_label_mat) ** 2
        numerator = torch.sum(weight * y_pred)
        label_dist = torch.sum(y_true, axis=0)
        pred_dist = torch.sum(y_pred, axis=0)
        w_pred_dist = torch.t(torch.matmul(weight_mat, pred_dist))
        denominator = torch.sum(torch.matmul(label_dist, w_pred_dist/batch_size),axis = 0)
        loss = torch.log(numerator/denominator + 1e-7)
 
        return (loss, outputs) if return_outputs else loss

class SOFT10Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.module.num_labels
        dist_matrix = model.module.dist_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-10*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-10*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class SOFT5Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.module.num_labels
        dist_matrix = model.module.dist_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-5*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-5*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class SOFT2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.module.num_labels
        dist_matrix = model.module.dist_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-2*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-2*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class SOFT3Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.module.num_labels
        dist_matrix = model.module.dist_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-3*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-3*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class SOFT4Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.module.num_labels
        dist_matrix = model.module.dist_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-4*dist_matrix[true_labels[j][i]][label_ids[j][i]])/np.sum([np.exp(-4*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs,device='cuda:0', requires_grad=True)
        err = -torch.log(probas)*softs_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss


class OLL1Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.module.num_labels
        dist_matrix = model.module.dist_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*distances_tensor
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class OLL15Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.module.num_labels
        dist_matrix = model.module.dist_matrix
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*distances_tensor**(1.5)
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

    
