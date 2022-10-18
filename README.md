# Natural Language Processing Project Summary

## Project Goals

> - Work on a NLP project that touches the full DS pipeline (data acquistion (web scraping), preparation, exploratory data analysis, modeling, and model evaluation), provide findings, and offer key takeaways.
> - Create modules (acquire.py, prepare.py, explore.py, and modeling.py) that make our process repeateable and our report (notebook) easier to read and follow.
> - Ask exploratory questions of our data that will help us understand more about how diction differs or is similar between programming languages.
> - Construct a model to predict the main programming language of a repository, given the text of the README file.


## Project Description

### Deliverables

> - **Readme (.md)**
> - **Prepare Modele (.py)**
> - **Final Notebook (.ipynb)**
> - **Recorded Presentation w/ slides**

## Data Dictionary

|Target|Definition
|:-------|:----------|
|language|the identified programming language used within the repository|

|Feature|Definition|
|:-------|:----------|
|repo        |the name of the repository|
|original    |the original form of the text found in the README file|
|clean       |the README text after special characters are removed|
|stemmed     |the README text after breaking down into segments and converting each word to its 'stem' word|
|lemmatized  |the README text after breaking down into segments and converting each word to its 'root' word| 


## Initial Hypotheses
-  _____________

## Executive Summary - Key Findings and Recommendations
> 1. __________________

> 2. Our recommendations are ________________

> 3. Next steps ____________________

## Project Plan

### Planning Phase

> - Created a README.md file
> - Imported all of our tools and files needed to properly conduct the project.

### Acquire Phase

> - Utilized our acquire file to web scrape the gituhub search results to gather our data.

### Prepare Phase

> - Utilized our prepare file to clean up the scraped README content.
> - Split the overall dataset into our train, validate, and test datasets.

### Explore Phase

> - Created visualizations via word clouds and bar plots to identify trends in word usage.
> - Asked clear questions about that data such as ____________.
> - Utilized hypothesis testing to best answer and provide insight into our aformentioned questions.

### Model Phase

> -  _________________________

### Deliver Phase

> - Prepped our final notebook with a clean presentation to best present our findings and process to our peers and instructors.

## How To Reproduce My Project

> 1. Read this README.md
> 2. Download the acquire.py, prepare.py, explore.py, modeling.py and final_report.ipynb files into your directory along with the languages.csv.
> 3. Enusre you have an env.py file that contains your github_token, and github_username.
> 4. Run our final_report.ipynb notebook
> 4. Congratulations! You can predict programming languages found within any repository on github using just a README file!

## Attachments/Links
