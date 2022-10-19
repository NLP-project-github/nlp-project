# from __future__ import division
import itertools

# To get rid of those blocks of red warnings
import warnings
warnings.filterwarnings("ignore")

# Standard Imports
import numpy as np
from scipy import stats
import pandas as pd
from math import sqrt
import os
from scipy.stats import spearmanr
from sklearn import metrics
from random import randint
from typing import Dict, List, Optional, Union, cast
from time import sleep

# Vis Imports
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import plotly.express as px
from pandas.plotting import register_matplotlib_converters
from mpl_toolkits.mplot3d import Axes3D
from wordcloud import WordCloud

# Modeling Imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import sklearn.preprocessing
import statsmodels.api as sm
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# NLP Imports
import unicodedata
import re
import json
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

def model_top_3(df):
    """
    This function runs our top 3 models on the train and validate sets,
    it then prints out the results.
    """
    # Make the object
    tfidf = TfidfVectorizer()
    # Fit/Transform
    X = tfidf.fit_transform(df.lemmatized)
    # What we are predicting
    y = df.language
    # Split X and y into train, validate, and test 
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=123)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=.25, random_state=123)

    # Make train and validate a dataframe
    train = pd.DataFrame(dict(actual=y_train))
    validate = pd.DataFrame(dict(actual=y_validate))
    # Make the object and fit it
    lm = LogisticRegression().fit(X_train, y_train)
    # Make columns for the predictions
    train['predicted_lm'] = lm.predict(X_train)
    validate['predicted_lm'] = lm.predict(X_validate)
    # Make the object and fit it
    MNBclf = MultinomialNB()
    MNBclf.fit(X_train, y_train)
    # Make columns for the predictions
    train['predicted_MNBclf'] = MNBclf.predict(X_train)
    validate['predicted_MNBclf'] = MNBclf.predict(X_validate)
    # Make the object and fit/transform it
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df.lemmatized)
    # Split X and y into train, validate, and test.
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=123)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=.25, random_state=123)
    # Make the object and fit it
    lm = LogisticRegression().fit(X_train, y_train)
    # Make columns for the predictions
    train['bow_predicted_lm'] = lm.predict(X_train)
    validate['bow_predicted_lm'] = lm.predict(X_validate)
    # Print out the results
    print('Bag of Words Logistic Regression Train Accuracy: {:.2%}'.format(accuracy_score(train.actual, train.bow_predicted_lm)))
    print('-------------')
    print('Bag of Words Logistic Regression Validate Accuracy: {:.2%}'.format(accuracy_score(validate.actual, validate.bow_predicted_lm)))
    print('-------------')
    print('TF-IDF MultinomialNB Train Accuracy: {:.2%}'.format(accuracy_score(train.actual, train.predicted_MNBclf)))
    print('-------------')
    print('TF-IDF MultinomialNB Validate Accuracy: {:.2%}'.format(accuracy_score(validate.actual, validate.predicted_MNBclf)))
    print('-------------')
    print('Best Model👇')
    print('-------------')
    print('TF-IDF Logistic Regression Train Accuracy: {:.2%}'.format(accuracy_score(train.actual, train.predicted_lm)))
    print('-------------')
    print(classification_report(train.actual, train.predicted_lm))
    print('-------------')
    print('TF-IDF Logistic Regression Validate Accuracy: {:.2%}'.format(accuracy_score(validate.actual, validate.predicted_lm)))
    print('-------------')
    print(classification_report(validate.actual, validate.predicted_lm))

def model_test(df):
    """
    This function uses the TF-IDF feature to run a 
    logistic regression model on the test dataset
    and print the results.
    """
    # Make the object
    tfidf = TfidfVectorizer()
    # Fit/Transform
    X = tfidf.fit_transform(df.lemmatized)
    # What we are predicting
    y = df.language
    # Split X and y into train, validate, and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=123)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=.25, random_state=123)
    # Make test a dataframe
    test = pd.DataFrame(dict(actual=y_test))
    # Make the object and fit it
    lm = LogisticRegression().fit(X_train, y_train)
    # Make a column for the predictions
    test['predicted_lm'] = lm.predict(X_test)
    # Print the results
    print('TF-IDF Logistic Regression Test Accuracy: {:.2%}'.format(accuracy_score(test.actual, test.predicted_lm)))
    print('-------------')
    print(classification_report(test.actual, test.predicted_lm))