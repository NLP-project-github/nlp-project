"""
A module for obtaining repo readme and language data from the github API.

Before using this module, read through it, and follow the instructions marked
TODO.

After doing so, run it like this:

    python acquire.py

To create the `data.json` file that contains the data.
"""

# common imports
import os
import pandas as pd
import numpy as np

# imports recommended by curriculumn 
from typing import Dict, List, Optional, Union, cast

# importing access information from custom env file
from env import github_token, github_username

# imports used to process NLP methodology
import requests
import json
from bs4 import BeautifulSoup
from time import sleep
from random import randint
# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

# Our data comes from the top 160 forked repositories on Github as of 17 Oct 2022 (languages.csv)


# web scraping by top four programming languages (JavaScript, Python, Java, C)
list_rep = []

for i in range(1,5):
    headers = {'User-Agent': github_username}
    sleep(randint(2,10))
    response = requests.get('https://github.com/search?l=JavaScript&o=desc&p=' + str(i) + '&q=stars%3A%3E1&s=forks&type=Repositories&spoken_language_code=en', headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    for repo in soup.find_all('a', class_ = 'v-align-middle'):
        list_rep.append(repo.text)

# Python pull        
list_rep2 = []

for i in range(1,5):
    headers = {'User-Agent': github_username}
    sleep(randint(2,10))
    response = requests.get('https://github.com/search?l=Python&o=desc&p=' + str(i) + '&q=stars%3A%3E1&s=forks&type=Repositories&spoken_language_code=en', headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    for repo in soup.find_all('a', class_ = 'v-align-middle'):
        list_rep2.append(repo.text)


# Java pull        
list_rep3 = []

for i in range(1,5):
    headers = {'User-Agent': github_username}
    sleep(randint(2,10))
    response = requests.get('https://github.com/search?l=Java&o=desc&p=' + str(i) + '&q=stars%3A%3E1&s=forks&type=Repositories&spoken_language_code=en', headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    for repo in soup.find_all('a', class_ = 'v-align-middle'):
        list_rep3.append(repo.text)


# C pull        
list_rep4 = []

for i in range(1,5):
    headers = {'User-Agent': github_username}
    sleep(randint(2,10))
    response = requests.get('https://github.com/search?l=C%2B%2B&o=desc&p=' + str(i) + '&q=stars%3A%3E1&s=forks&type=Repositories&spoken_language_code=en', headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    for repo in soup.find_all('a', class_ = 'v-align-middle'):
        list_rep4.append(repo.text)
        

# combines sepearate pulls of repo names
full_list = list_rep + list_rep2 + list_rep3 + list_rep4
        
# renames to REPOS to be used later     
REPOS = full_list

# assigns token and username from env.py
headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    
    '''
    Given an url and returns a response code 
    '''
    
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    
    '''
    Given a repository name and returns the language tagged in the repository 
    '''
    
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    
    '''
    Given a repository name and returns content found within the repository. 
    '''
    
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)
