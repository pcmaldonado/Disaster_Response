# Overview
As part of the <a href="https://www.udacity.com/course/data-scientist-nanodegree--nd025">Udacity Data Scientist Nanodegree Program</a>, this multioutput classification project aims to analyze and classify messages to improve communication during disasters, using data provided by <a href="https://appen.com/">Appen</a> (formally Figure 8) that contains real messages that were sent during disaster events.

A web application hosted on [Heroku]() allows to input new messages for classification.

# Directories

## Config
To avoid errors and increase efficiency, configuration setup is handled in the **"config"** directory to avoid input user when running the scripts. Thus, users are not expected to write name files when running the python script. Instead, the *core.py* and the *config.yml* make sure data is as expected.

To re-train the model, e.g. if new data becomes available, the *"config.yml"* should be updated with the new file names, or the previous data files should be replaced.

## Data
It contains the raw data (messages and categories) in csv files, as well as the cleaned data in a database (.db) file.

It also contains the python script needed to apply the entire ETL process, process_data.py, which extracts data from csv files, transforms them and then loads them into a single SQLite database.

To run this script on the command line, from the project folder:
`python data/process_data.py`

## Models
It contains the python script that handles all the machine learning steps needed for this project, *train_classifier.py*, as well as an additional script called *functions.py* that provides functions for the web application, as well as a single shared function for *train_classifier.py* : "tokenize". This is done mainly to avoid problems when deploying on the live app.

The directory also holds the pickle file containing the best model from the GridSearchCV done on the training set.

To run the main script, *train_classifier.py* from the command line, on the project folder run this script: `python models/train_classifier.py`

## App
It contains the scripts and necessary files to run the wep application.

To access the web application on a local computer, run: `python app/run.py` and run the given url.
