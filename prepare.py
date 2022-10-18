# standard imports
import pandas as pd
import numpy as np
import os

# import custom files
import acquire

# imports used to manipulate text
import unicodedata
import re
import json
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk. corpus import stopwords


# creates initial dataframe either from csv or running acquire.py to web scrape data directly
def pull_data():
    
    '''
    Returns initial dataframe with data pulled from github.com
    '''
    
    filename = "languages.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        execfile('acquire.py')
        df = open('data.json')
        df = json.load(df)
        df = pd.DataFrame(df)
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df 



def basic_clean(original):
    
    '''
    Takes in a dataframe and returns a dataframe with standardized syntax.
    '''
    # removing null values
    original = original.dropna()
    # changes all words to lowercase
    article = original.lower()
    # removes any oddities in unicode character encoding
    article = unicodedata.normalize('NFKD', article)\
    .encode('ascii', 'ignore')\
    .decode('utf-8')
    #use re.sub to remove special characters
    article = re.sub(r'[^a-z\s]', '', article)
    
    return article

def tokenize(article):
    
    '''
    Takes in cleaned dataframe and returns a dataframe with text broken down from larger body
    to smaller bodies of text.
    '''
    
    #create the tokenizer
    tokenize = nltk.tokenize.ToktokTokenizer()
    #use the tokenizer
    article = tokenize.tokenize(article, return_str=True)

    return article

def stem(article):
    
    '''
    Takes in a dataframe and returns a dataframe with its text altered to its base form for each
    word.
    '''
    
    #create porter stemmer
    ps = nltk.porter.PorterStemmer()
    # apply the porter stemmer to each word in the series
    stems = [ps.stem(word) for word in article.split()]
    #join words back together
    article_stemmed = ' '.join(stems)
    
    return article_stemmed

def lemmatize(article):
    
    '''
    Takes in a dataframe and returns a dataframe with its text altered to its base form for each word
    (root words).
    '''
    
    #create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    # apply the lemmatizer to each word in the series
    lemmas = [wnl.lemmatize(word) for word in article.split()]
    #join words back together
    article_lemmatized = ' '.join(lemmas)
    
    return article_lemmatized

def remove_stopwords(article):
    
    '''
    Takes in a dataframe and removes all any words identified as a stopword and returns new dataframe.
    '''
    
    #save stopwords
    stopwords_ls = stopwords.words('english')
    words = article.split()
    #remove stopwords from list of words
    filtered_words = [word for word in words if word not in stopwords_ls]
    #join words back together
    article = ' '.join(filtered_words)
    
    return article

def prepare_article(original):
    
    '''
    Takes in a dataframe and applies various other functions to create new columns for clean, stemmed,
    and lemmatized data.  Returns new dataframe with updated columns
    '''
    
    # creates cleaned readme_content column
    original['clean'] = original['readme_contents'].apply(basic_clean).apply(tokenize).apply(remove_stopwords)
    # creates stemmed readme_content column
    original['stemmed'] = original['readme_contents'].apply(basic_clean).apply(tokenize).apply(remove_stopwords).apply(stem)
    # creates lemmatized readme_content column
    original['lemmatized'] = original['readme_contents'].apply(basic_clean).apply(tokenize).apply(remove_stopwords).apply(lemmatize)
    # renames readme_content column as original 
    original.rename(columns = {'readme_contents':'original'}, inplace = True)
    
    return original