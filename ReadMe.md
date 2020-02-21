# Introduction

This project is based off the first part of the [Fake News Challenge](http://www.fakenewschallenge.org/)(FNC-1) where the goal is "to explore how AI...might be able to combat the fake news problem."  Identifying fake news is complex task that can be broken down into a few steps, with a potential first being the comparison of topics or, more precisely, **stance detection**, across myriad news organizations.  The goal of this project is to classify the relationship between a body of text with a headline as agree, disagree, discuss or unrelated.   

# Modeling

The **preprocessing** work here includes the removal of punctuation, lower casing all text, the removal of stop words (as the goal is classification), tokenizing and stemming.  

**Feature engineering is inclusive of:**
 - basic n_gram count ratios
 - TF-IDF 
 - SVD
 - word embeddings with Word2Vec using the [Google News Corpus pre-trained weights](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
 - sentiment features to assign polarity, using [nltk Sentiment Analyzer](https://www.nltk.org/_modules/nltk/sentiment/vader.html) with [VaderSentiment](https://github.com/mgavish/vaderSentiment) 
 
 
**Classification** is completed with XGBoost (Gradient Boosting) and a CNN.

# Dependancies

- Scipy (pandas, numpy, matplotlib)
- Scikit-Learn
- NLTK
- Gensim
- Keras

I completed all work on a Google Compute Engine (GCE) VM.  You can get up and running quickly with a jupyter connected VM on GCE following my post on [Medium](https://medium.com/@mngavish/deep-learning-on-google-compute-engine-through-jupyter-interface-15d64e7d7e00)  
Executing the preprocessing and feature engineering in one session will need ~800GB though, all data will be pickled for later use. I would however advise using numpy's .npz instead as it has quicker read/write times.  


# Future Work
 - DL with Conv1D