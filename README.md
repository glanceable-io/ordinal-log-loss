# Ordinal Log Loss - A simple loss function for Ordinal Classification

This is the GitHub repository for the paper: 
A simple log-based loss function for ordinal text classification

### Paper Abstract
The cross-entropy loss function is widely used and generally considered the default loss function for text classification. When it comes to ordinal text classification where there is an ordinal relationship between labels, the cross-entropy is not optimal as it does not incorporate the ordinal character into its feedback. In this paper, we propose a new simple loss function called ordinal log-loss (OLL). We show that this loss function outperforms state-of-the-art previously introduced losses on four benchmark text classification datasets. 


**This repository contains all the python code used to conduct the experiments reported in the paper.**

---

## Datasets 

The experiments were done on 4 public datasets : 
* **SNLI** (Stanford Natural Language Inference): The Stanford Natural Language Inference (SNLI) corpus (version 1.0) is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral. We aim for it to serve both as a benchmark for evaluating representational systems for text, especially including those induced by representation-learning methods, as well as a resource for developing NLP models of any kind. (source : https://nlp.stanford.edu/projects/snli/)
* **SST-5** : Sentiment classification of sentences extracted from movie reviews. Each sentence is labelled as either negative, somewhat negative, neutral, somewhat positive or positive.  (source : https://nlp.stanford.edu/sentiment/)
* **Amazon Reviews** : Sentiment classification of customer reviews on the Amazon website. Each sentence is labelled as either negative, somewhat negative, neutral, somewhat positive or positive.  (source : https://registry.opendata.aws/amazon-reviews-ml/)
***Yelp Reviews** : Sentiment classification of sentences extracted from the Yelp website. Each sentence is labelled as either negative, somewhat negative, neutral, somewhat positive or positive.  (source : https://www.yelp.com/dataset)


## Training

### Pre-trained model
The model used in our experiments is the (https://huggingface.co/google/bert_uncased_L-2_H-128_A-2)[google/bert_uncased_L-2_H-128_A-2] which is a tiny version of the BERT model. This model can be fetched directly from the HuggingFace Model Hub.

### Losses 

