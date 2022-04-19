# Ordinal Log Loss - A simple loss function for Ordinal Classification

This is the GitHub repository for the paper: 
A simple log-based loss function for ordinal text classification

## Paper Abstract
The cross-entropy loss function is widely used and generally considered the default loss function for text classification. When it comes to ordinal text classification where there is an ordinal relationship between labels, the cross-entropy is not optimal as it does not incorporate the ordinal character into its feedback. In this paper, we propose a new simple loss function called ordinal log-loss (OLL). We show that this loss function outperforms state-of-the-art previously introduced losses on four benchmark text classification datasets. 


**This repository contains all the python code used to conduct the experiments reported in the paper.**

---
## Losses

In the paper, we introduce a new loss called the Ordinal Log Loss (OLL). We show that this loss, in addition to being very simple, is particularly suited for classification tasks where labels are more or less close to each other (e.g. Movie review rating classification). 

For a N classes classification task, we define the L<sub>OLL-&alpha;</sub> loss (with &alpha; is a tuneable parameter)

<img src="https://render.githubusercontent.com/render/math?math=\Large\color{grey}\textbf{\mathcal{L}_{OLL-\alpha}(P,y) = -\sum_{i=1}^{N}\log(1-p_i) d(y,i)^\alpha}">
where P = (p<sub>1</sub>, ..., p<sub>N</sub> ) is the output probability distribution of a network for a given prediction and d(y,i) is the distance between the true class y and the class i.

We compare this loss to 5 other losses :
* [Cross Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) (CE)
* [Weighted Kappa Loss](https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666?via%3Dihub) (WKL)
* [Soft Labels Loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html) (SOFT)
* [Earth Mover Distance Loss](https://arxiv.org/abs/1611.05916) (EMD)
* [Coral Loss](https://github.com/Raschka-research-group/coral-cnn) (CORAL)

The losses used in the experiments have been coded in pytorch and can be found in the file : `src/loss_functions.py`

## Datasets 

The experiments were done on 4 public datasets : 
* **[SNLI](https://nlp.stanford.edu/projects/snli/)** (Stanford Natural Language Inference): The Stanford Natural Language Inference (SNLI) corpus (version 1.0) is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral. 
* **[SST-5](https://nlp.stanford.edu/sentiment/)** : Sentiment classification of sentences extracted from movie reviews. Each sentence is labelled as either negative, somewhat negative, neutral, somewhat positive or positive.
* **[Amazon Reviews](https://registry.opendata.aws/amazon-reviews-ml/)** : Sentiment classification of customer reviews on the Amazon website. Each sentence is labelled as either negative, somewhat negative, neutral, somewhat positive or positive.
* **[Yelp Reviews](https://www.yelp.com/dataset)** : Sentiment classification of sentences extracted from the Yelp website. Each sentence is labelled as either negative, somewhat negative, neutral, somewhat positive or positive.


## Training

### Pre-trained model
The model used in our experiments is the [google/bert_uncased_L-2_H-128_A-2](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) which is a tiny version of the BERT model. This model can be fetched directly from the HuggingFace Model Hub.

### Reproducibility
All our experiments can be reproduced. To do so, follow steps:

1. **Download datasets**

All the datasets mentioned above can be downloaded from the [Hugging Face Datasets Hub](https://huggingface.co/datasets). 
Once downloaded, edit the `src/datasets.json` with the corresponding path for each dataset:
```
"amazon_reviews": {
        "dist": [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]], 
        "int2label": [1, 2, 3, 4, 5], 
        "num_classes": 5, 
        "task": ["text", null], 
        "tok_len": 128,
        "n_distances": 5,
        "path" : "COMPLETE WITH THE PATH OF DIRECTORY WITH NAME 'amazon_reviews' CONTAINING TRAIN,VALIDATION AND TEST FILES"
    }
```
For the *Amazon Reviews Dataset*, you should have a folder named `amazon_reviews` with in it the 3 following files: `amazon_reviews_train.csv`, `amazon_reviews_test.csv` and `amazon_reviews_validation.csv`.

2. Training

To train the model on the different parameters and loss functions introduced in our paper, run the `src/training.py` Python script. 

The losses used for training: 
```
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
```
are defined in the `src/loss_functions.py` file. 

3. Evaluation

Run the `scr/inference.py` file to evaluate the the model checkpoints generated during the training phase (corresponding to the different losses and parameters). It will output a csv file `src/outputs_training/output_metrics/metrics_test_set.csv` with all metrics introduced in our paper on the test sets. 

**Note**: In `src/model_coral.py` we reimplemented the coral method as presented [here](https://github.com/Raschka-research-group/coral-cnn). 


