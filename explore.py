import pandas as pd
import numpy as np

from requests import get
from bs4 import BeautifulSoup
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
import nltk.sentiment
import re
import prepare
from wordcloud import WordCloud

# cleanup function
def clean(text: str, ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt']) -> list:
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
        text = (text.encode('ascii', 'ignore')
                .decode('utf-8', 'ignore')
                .lower())
        words = re.sub(r'[^\w\s]', '', text).split() 
        return [wnl.lemmatize(word) for word in words if word not in stopwords]

# this function returns the most common word
def most_frequent_word(s: pd.Series) -> str:
    words = clean(' '.join(s))
    most_common_word = pd.Series(words).value_counts().head(1).index
    return most_common_word

#this function returns the most common bigram
def most_frequent_bigram(s: pd.Series) -> str:
    words = clean(' '.join(s))
    most_common_bigram = pd.Series(nltk.bigrams(words)).value_counts().head(1).index
    return most_common_bigram

#this function cleans up most common words
def most_common_words(df):

    ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt']
    def clean(text: str) -> list:
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
        text = (text.encode('ascii', 'ignore')
                .decode('utf-8', 'ignore')
                .lower())
        words = re.sub(r'[^\w\s]', '', text).split() # tokenization
        return [wnl.lemmatize(word) for word in words if word not in stopwords]

    all_words = clean(' '.join(df.clean_lemmatized))
    python_words = clean(' '.join(df[df.language == 'Python'].clean_lemmatized))
    javascript_words = clean(' '.join(df[df.language == 'JavaScript'].clean_lemmatized))
    java_words = clean(' '.join(df[df.language == 'Java'].clean_lemmatized))
    c_plus_plus_words = clean(' '.join(df[df.language == 'C++'].clean_lemmatized))
    c="#f7965cff"
    figure, axes = plt.subplots(1, 6)
    
    pd.Series(python_words).value_counts().head(12).plot.barh(width=.9, ec='black',color = c, title='12 most common words in Python', figsize=(19,10), ax=axes[1])
    pd.Series(javascript_words).value_counts().head(12).plot.barh(width=.9, ec='black',color = c, title='12 most common words in JavaScript', figsize=(19,10), ax=axes[2])
    pd.Series(java_words).value_counts().head(12).plot.barh(width=.9, ec='black',color = c, title='12 most common words Java', figsize=(19,10), ax=axes[3])
    pd.Series(c_plus_plus_words).value_counts().head(12).plot.barh(width=.9, ec='black',color = c, title='12 most common words C++', figsize=(19,10), ax=axes[4])
    
    plt.tight_layout()

#function to visualize word count distribution
def plot_word_count_distribution(df):
    
    def word_count(word):
        word_count = len(re.findall(r'\w+', word))
        return word_count

    word_count = df.clean_lemmatized.apply(word_count)
    df["word_count"] = word_count

    plt.figure(figsize=(12,8))
    sns.distplot(df.word_count, color="#f7965cff")
    plt.title("Word Count Distribution")
    
 #this function is for visualizing value counts of readmes of all languages in a bar graph   
def plot_distro_for_value_counts_all(df):
    
    c="#f7965cff"
    value_counts_all = pd.DataFrame(df.language.value_counts(ascending=False))
    plt.figure(figsize=(13,10))
    bar = sns.barplot(x=value_counts_all.index, y="language", data=value_counts_all, color = c)
    bar.set_xticklabels(bar.get_xticklabels(),rotation=65)
    bar.set_ylabel("counts")

    plt.title("Visualize how is data distributed per document for all languages")
    plt.show()
    
 #this function is for visualizing value counts of readmes of top languages in a bar graph   
def plot_distro_for_value_counts_top(df):
    c="#f7965cff"
    value_counts = pd.DataFrame(df.is_top_language.value_counts(ascending=False))
    plt.figure(figsize=(13,10))
    bar = sns.barplot(x=value_counts.index, y="is_top_language", data=value_counts, color = c)
    bar.set_xticklabels(bar.get_xticklabels(),rotation=65)
    bar.set_ylabel("counts")

    plt.title("Visualize how is the data distributed per document for top languages")
    plt.show()
    
#this function is for visualizing readmes & their lengths in a scatterplot    
def scatterplot_for_readmes(df):
    df_length = df.assign(length = df.clean_lemmatized.apply(len))

    plt.figure(figsize=(13,10))
    ax = plt.subplot(111)

    plt.title("Visualize lengths of readme files per programming language")
    sns.scatterplot(y=df_length.length, x=df_length.index,hue=df_length.language)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)
    plt.show()
    
 #function to visualize readme counts on a bargraph showing the min, max and median   
def bargraphs_for_min_max_median(df):
    
    df_length = df.assign(length = df.clean_lemmatized.apply(len))

    median_lengths = df_length.groupby("language").median().sort_values(by="length", ascending= False)
    max_length = pd.DataFrame(df_length.groupby("language").length.max().sort_values(ascending= False))
    min_length = pd.DataFrame(df_length.groupby("language").length.min().sort_values(ascending= False))

    c="#f7965cff"
    plt.figure(figsize=(13,10))
    plt.title("Visualize median length of readme files per language")
    bar = sns.barplot(y=median_lengths.length,x=median_lengths.index, color=c)

    bar.set_xticklabels(bar.get_xticklabels(),rotation=65)
    plt.show()

    plt.figure(figsize=(13,10))
    plt.title("Visualize minimum length of readme files per language")
    bar = sns.barplot(y=min_length.length,x=min_length.index, color=c)

    bar.set_xticklabels(bar.get_xticklabels(),rotation=65)
    plt.show()

    plt.figure(figsize=(13,10))
    plt.title("Visualize maximum length of readme files per language?")
    bar = sns.barplot(y=max_length.length,x=max_length.index, color=c)

    bar.set_xticklabels(bar.get_xticklabels(),rotation=65)
    plt.show()

 #function checking on word count of readme   
def word_count(word):
    word_count = len(re.findall(r'\w+', word))
    return word_count

#function checking on digit count of readme 
def digit_count(word):
    digit_count = len(re.findall(r'\d+', word))
    return digit_count

#this function returns a summary of the top language word counts    
def word_count_summary(df):
    df["word_count"] = df.clean_lemmatized.apply(word_count)
    
    min_word_count = pd.DataFrame(df.groupby("is_top_language").word_count.min())
    min_word_count.columns = ['Min Word Count']

    max_word_count = pd.DataFrame(df.groupby("is_top_language").word_count.max())
    max_word_count.columns = ["Max Word Count"]

    median_word_count = pd.DataFrame(df.groupby("is_top_language").word_count.median())
    median_word_count.columns = ["Median Word Count"]

    mean_word_count = pd.DataFrame(df.groupby("is_top_language").word_count.mean())
    mean_word_count.columns = ["Mean Word Count"]

    std_word_count = pd.DataFrame(df.groupby("is_top_language").word_count.std())
    std_word_count.columns = ["STD of Word Count"]
    
    summary1 = pd.merge(min_word_count, max_word_count, left_index=True, right_index=True)
    summary2 = pd.merge(median_word_count , mean_word_count , left_index=True, right_index=True)
    summary3 = pd.merge(summary1 , summary2 , left_index=True, right_index=True)
    summary = pd.merge(summary3 , std_word_count , left_index=True, right_index=True)
    
    return summary

#This function lists words for each language and any other language 
def list_of_words_for_top_languages(dfx, language="is_top_language", cleaned="clean_lemmatized"):
    # create list of words by language

    js_words = ' '.join(dfx[dfx[language] == 'JavaScript'][cleaned]).split()
    p_words = ' '.join(dfx[dfx[language] == 'Python'][cleaned]).split()
    j_words = ' '.join(dfx[dfx[language] == 'Java'][cleaned]).split()
    cpp_words = ' '.join(dfx[dfx[language] == 'C++'][cleaned]).split()
    all_words = ' '.join(dfx[cleaned]).split()
    
    return js_words, p_words, j_words, cpp_words, all_words


#this function returns a series of top 20 bigrams all languages
def create_bigrams(df):
    
    javascript_words, python_words, java_words, cpp_words, all_words = list_of_words_for_top_languages(df, language="is_top_language", cleaned="clean_lemmatized")

    top_20_bigrams = (pd.Series(nltk.ngrams(all_words, 3))
                      .value_counts()
                      .head(20))

    top_20_js_bigrams = (pd.Series(nltk.ngrams(javascript_words, 3))
                      .value_counts()
                      .head(20))

    top_20_p_bigrams = (pd.Series(nltk.ngrams(python_words, 3))
                      .value_counts()
                      .head(20))

    top_20_j_bigrams = (pd.Series(nltk.ngrams(java_words, 3))
                      .value_counts()
                      .head(20))

    top_20_cpp_bigrams = (pd.Series(nltk.ngrams(cpp_words, 3))
                      .value_counts()
                      .head(20))

    return top_20_bigrams, top_20_js_bigrams, top_20_p_bigrams, top_20_j_bigrams, top_20_cpp_bigrams

#this function plots bigrams for top 20 of all the languages
def plot_bigrams(df):
    
    top_20_bigrams, top_20_js_bigrams, top_20_p_bigrams, top_20_j_bigrams, top_20_cpp_bigrams= create_bigrams(df)
    
    c="#f7965cff"

    top_20_js_bigrams.sort_values().plot.barh(color=c, width=.9, figsize=(13, 10))
    plt.title('Most Frequently Occuring Java Script Bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurences')
    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_js_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    plt.show()


    top_20_p_bigrams.sort_values().plot.barh(color=c, width=.9, figsize=(13, 10))
    plt.title('Most Frequently Occuring Python Bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurences')
    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_p_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    plt.show()

    top_20_j_bigrams.sort_values().plot.barh(color=c, width=.9, figsize=(13,10))
    plt.title('Most Frequently Occuring Java Bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurences')
    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_j_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    plt.show()

    top_20_cpp_bigrams.sort_values().plot.barh(color=c, width=.9, figsize=(13,10))
    plt.title('Most Frequently Occuring C++ Bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurences')
    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_cpp_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    plt.show()

    
 #this function creates a word cloud for all the languages   
def word_cloud(text):
    plt.figure(figsize=(13, 13))

    cloud = WordCloud(background_color='white', height=1000, width=1000).generate(' '.join(text))

    plt.imshow(cloud)
    plt.axis('off')



