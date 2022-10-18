# To get rid of those blocks of red warnings
import warnings
warnings.filterwarnings("ignore")

# Standard Imports
import numpy as np
from scipy import stats
import pandas as pd
import os
from scipy.stats import spearmanr
from sklearn import metrics
from random import randint
from typing import Dict, List, Optional, Union, cast
from time import sleep

# Vis Imports
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Modeling Imports
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

# NLP Imports
import unicodedata
import re
import json
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

def q1(JavaScript_words, Python_words, Java_words, C_plus_plus_words, all_words):
    """
    This function outputs a dataframe of the frequency of a word in each coding language.
    """
    # Makes it a series and performs a value count
    JavaScript_freq = pd.Series(JavaScript_words).value_counts()
    Python_freq = pd.Series(Python_words).value_counts()
    Java_freq = pd.Series(Java_words).value_counts()
    C_plus_plus_freq = pd.Series(C_plus_plus_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()
    # Creates a dataframe to view the info easier
    word_counts = (pd.concat([all_freq, JavaScript_freq, Python_freq, Java_freq, C_plus_plus_freq], axis=1, sort=True)
                .set_axis(['all', 'javascript', 'python', 'java', 'c_plus_plus'], axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int)))
    
    return word_counts

def q2_word_cloud_1(JavaScript_words, Python_words, Java_words, C_plus_plus_words, all_words):
    all_cloud = WordCloud(background_color='white', height=1000, width=400).generate(' '.join(all_words))
    javascript_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(JavaScript_words))
    python_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(Python_words))

    plt.figure(figsize=(10, 8))
    axs = [plt.axes([0, 0, .5, 1]), plt.axes([.5, .5, .5, .5]), plt.axes([.5, 0, .5, .5])]

    axs[0].imshow(all_cloud)
    axs[1].imshow(javascript_cloud)
    axs[2].imshow(python_cloud)

    axs[0].set_title('All Words')
    axs[1].set_title('Javascript')
    axs[2].set_title('Python')

    for ax in axs: ax.axis('off')

def q2_word_cloud_2(JavaScript_words, Python_words, Java_words, C_plus_plus_words, all_words):
    all_cloud = WordCloud(background_color='white', height=1000, width=400).generate(' '.join(all_words))
    java_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(Java_words))
    c_plus_plus_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(C_plus_plus_words))

    plt.figure(figsize=(10, 8))
    axs = [plt.axes([0, 0, .5, 1]), plt.axes([.5, .5, .5, .5]), plt.axes([.5, 0, .5, .5])]

    axs[0].imshow(all_cloud)
    axs[1].imshow(java_cloud)
    axs[2].imshow(c_plus_plus_cloud)

    axs[0].set_title('All Words')
    axs[1].set_title('Java')
    axs[2].set_title('C++')

    for ax in axs: ax.axis('off')