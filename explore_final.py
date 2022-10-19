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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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

def q3(JavaScript_words, Python_words, Java_words, C_plus_plus_words, all_words):
    # Making Bigrams for each coding language
    top_20_javascript_bigrams = (pd.Series(nltk.ngrams(JavaScript_words, 2))
                      .value_counts()
                      .head(20))
    top_20_python_bigrams = (pd.Series(nltk.ngrams(Python_words, 2))
                        .value_counts()
                        .head(20))
    top_20_java_bigrams = (pd.Series(nltk.ngrams(Java_words, 2))
                        .value_counts()
                        .head(20))
    top_20_c_plus_plus_bigrams = (pd.Series(nltk.ngrams(C_plus_plus_words, 2))
                        .value_counts()
                        .head(20))
    top_20_all_words_bigrams = (pd.Series(nltk.ngrams(all_words, 2))
                        .value_counts()
                        .head(20))
    
    # We can supply our own values to be used to determine how big the words (or
    # phrases) should be through the `generate_from_frequencies` method. The
    # supplied values must be in the form of a dictionary where the keys are the
    # words (phrases), and the values are numbers that correspond to the sizes.
    #
    # We'll convert our series to a dictionary, and convert the tuples that make up
    # the index into a single string that holds each phrase.

    data = {k[0] + ' ' + k[1]: v for k, v in top_20_all_words_bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=4000, height=2000).generate_from_frequencies(data)

    data2 = {k[0] + ' ' + k[1]: v for k, v in top_20_javascript_bigrams.to_dict().items()}
    img2 = WordCloud(background_color='white', width=2000, height=1000).generate_from_frequencies(data2)

    data3 = {k[0] + ' ' + k[1]: v for k, v in top_20_python_bigrams.to_dict().items()}
    img3 = WordCloud(background_color='white', width=2000, height=1000).generate_from_frequencies(data3)

    data4 = {k[0] + ' ' + k[1]: v for k, v in top_20_java_bigrams.to_dict().items()}
    img4 = WordCloud(background_color='white', width=2000, height=1000).generate_from_frequencies(data4)

    data5 = {k[0] + ' ' + k[1]: v for k, v in top_20_c_plus_plus_bigrams.to_dict().items()}
    img5 = WordCloud(background_color='white', width=2000, height=1000).generate_from_frequencies(data5)

    axs = [plt.axes([.5, 0, .5, .5]), plt.axes([0, 0, .5, .5]), plt.axes([.5, .5, .5, .5])
        , plt.axes([0, .5, .5, .5]), plt.axes([0, 1, 1, 1])]

    axs[4].imshow(img)
    axs[3].imshow(img2)
    axs[2].imshow(img3)
    axs[1].imshow(img4)
    axs[0].imshow(img5)
    axs[4].set_title('All Words')
    axs[3].set_title('Javascript')
    axs[2].set_title('Python')
    axs[1].set_title('Java')
    axs[0].set_title('C++')
    for ax in axs: ax.axis('off')
    plt.show()