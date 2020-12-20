# Multi-label classification on text
## Introduction
This repo presents several approaches tested in order to perform Multi-Label classification on
a data set made of Cross-Validated (branch of Stack-Overflow devoted to Machine Learning
and Statistics topics) posts. The goal is to predict the tags added by the creators of the
website. I did it in 2020. I used only the content of the posts and not the titles. The data set was found
on Kaggle at this address:

https://www.kaggle.com/stackoverflow/statsquestions

My personal goals were
* Get experience with implementing different algorithms
* Get an idea of what is achievable in terms of accuracy for mutli-label tasks with many
labels (several hundreds).
* See how different algorithms perform with respect to each other.  
  

## Data set information
The main reason why I picked it is because of its large number of labels: 1315 different
labels.
There are 85085 different samples, which leaves the door open for deep learning (even
if it is a bit small).
The distribution of labels is extremely
unbalanced: the most common label is "r" assigned to 13'236 samples. The least
common labels are assigned to only 1 sample.

## Algorithms tested
Here I outline the big categories of models I tried out in this project:
* Bag-of-Words representation and binary relevance based on logistic regeression
* Doc2Vec embedding representation and binary relevance based on logistic regeression 
* Trainable word embeddings followed by 1D Convolutional layer, dense layer, and sigmoid
activations
* Bag-of-word and Scikit Multi-Learn MLARAM
* Frozen pretrained BERT layer followed by dense layer and sigmoid activations

## Online resources used to build models
Here I list some links towards resources where I found some implementations or some 
ideas that I could (partially) reuse:
* Coursera course on NLP by the National Reaserch University Higher School
  of Economics of Moscow, see this link:
  https://www.coursera.org/learn/language-processing
  They have one notebook for multi-label classification on a similar (but smaller)
  dataset where they use bag-of-words and binary relevance with logistic regression.
* This tutorial:
  https://medium.com/towards-artificial-intelligence/multi-label-text-classification-using-scikit-multilearn-case-study-with-stackoverflow-questions-768cb487ad12
  where the author uses Doc2Vec and binary relevance (with Random Forest)
* This GitHub repo:
  https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb
  containing a notebook presenting the usage of a pretrained BERT model for
  a multi-label task (toxic comments) on text.
* Scikit Multi-learn which seems to be an extension of scikit learn specially
  designed for multi-label and multi-class problems
  http://scikit.ml/
* This notebook was the main inspiration source for the convolutional neural
  network model and its implementation:
  https://www.kaggle.com/roccoli/multi-label-classification-with-keras
  

## Findings and conclusions
* In order to compare the results I used a 80%, 10%, 10% split of the data
  into train, test, valid (but at the end I only used test for test). See
  data_splitting.py
* The only algorithm achieving somehow decent results (accuracy > 1% and macro
  F1 > 10%) was Bag-of-Words followed by binary relevance. This could be explained
  by the fact that the complexity of the task resides in the large number of labels
  and their imbalance, not in the complexity of the natural language. Most likely
  it is possible to predict labels simply by spotting a few keywords related to
  each label, which bag-of-word followed by binary relevance classifier based
  on logistic regression is suited for. The results were
  
  | Metric     |  Result   |
  :----------- |  -----------: |
  | accuracy   |  0.0412  |
  | F1 score macro |  0.1184 |
  F1 score micro |  0.3807 | 
  F1 score weighted |  0.3493 |
  Recall score |  0.0931 |
  Hamming loss | 0.0020 |
  Jaccard score | 0.0777 |
  
* For detailed description of the results for each pipeline (metrics + computational 
  time) see results_on_cross_validated.pdf. 
* I read several tutorials which were wrongly interpreting the Hamming loss they
  were obtaining for this dataset by pointing out how low it was.
  This one is a good example:
  https://medium.com/towards-artificial-intelligence/multi-label-text-classification-using-scikit-multilearn-case-study-with-stackoverflow-questions-768cb487ad12
  where the author gets a Hamming loss of 0.00219 and writes
  "That means almost 0.02% loss or we can say 99.98% accuracy !!"
  The problem is that the Hamming loss is the average over all the samples
  of the number of errors of prediction of presence or absence for each labels
  divided by the total number of labels. The problem with this is that given
  the fact that we have a huge number of labels (1'315) and that each post
  on stack overflow can have at most 5 tags, a model predicting for each post
  that no labels were assigned to it would have a Hamming loss of 5/1315 = 0.0038.
  Now if we take in account the fact that most likely, not all posts on
  Cross-Validated (Stack Overflow) have 5 tags but often less, we see that the
  result obtained in this tutorial is meaningless.
* I tried algorithms of Scikit Multi-learn but my computer often ran into memory
  shortage and for the ones which worked, it took a long time (unfortunately I
  didn't record it).
* The pipeline based on pretrained BERT was quickly trained (127.92 seconds)
  but prediction took almost an hour (3227.07)
* I guess that the difference between the micro and the macro averaging score
  for the best model is due to the imbalance between the tags. The most
  common tags account for most of the actual tags. So if we average the metric
  by tag (macro averaging) one uncommon tag weights as much in the balance as a common one.
  I guess that the models only learned how to correctly predict the common tags.
  But if all sample labels (by the y_ij where y_ij is 1 if sample i has tag j, 0
  otherwise) have the same weight (micro averaging), then focusing on common
  tags may yield a higher score.
* I tried to fight against imbalance data by restricting to the most common
  labels (but still including omitted uncommon ones in the computation
  of the metrics by using padding on the binary labals), in an effort to help
  models focusing on sufficiently common labels. I wrote a script (label_selection.py)
  where given a threshold of say 25, kept only the labels for which there
  were more than 25 samples having this label in the whole training set.
  
  
* I definitely need to learn to use online cloud computing resources. I tried
  Google COLAB but the resources are too limited and the scikit multi-learn
  based notebook also crashed. I tried to make a Google Cloud Computing account
  but it didn't accept my credit card because it is prepaid...




