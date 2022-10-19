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
    # Make the object
    tfidf = TfidfVectorizer()
    # Fit/Transform
    X = tfidf.fit_transform(df.lemmatized)
    # What we are predicting
    y = df.language

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=123)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=.25, random_state=123)

    train = pd.DataFrame(dict(actual=y_train))
    validate = pd.DataFrame(dict(actual=y_validate))

    lm = LogisticRegression().fit(X_train, y_train)

    train['predicted_lm'] = lm.predict(X_train)
    validate['predicted_lm'] = lm.predict(X_validate)

    MNBclf = MultinomialNB()
    MNBclf.fit(X_train, y_train)

    train['predicted_MNBclf'] = MNBclf.predict(X_train)
    validate['predicted_MNBclf'] = MNBclf.predict(X_validate)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df.lemmatized)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=123)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=.25, random_state=123)

    lm = LogisticRegression().fit(X_train, y_train)

    train['bow_predicted_lm'] = lm.predict(X_train)
    validate['bow_predicted_lm'] = lm.predict(X_validate)

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
    tfidf = TfidfVectorizer()
    # Fit/Transform
    X = tfidf.fit_transform(df.lemmatized)
    # What we are predicting
    y = df.language

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=123)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=.25, random_state=123)

    test = pd.DataFrame(dict(actual=y_test))

    lm = LogisticRegression().fit(X_train, y_train)

    test['predicted_lm'] = lm.predict(X_test)

    print('TF-IDF Logistic Regression Test Accuracy: {:.2%}'.format(accuracy_score(test.actual, test.predicted_lm)))
    print('-------------')
    print(classification_report(test.actual, test.predicted_lm))