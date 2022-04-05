# Disaster Response Project
# Overview
As part of the <a href="https://www.udacity.com/course/data-scientist-nanodegree--nd025">Udacity Data Scientist Nanodegree Program</a>, this multioutput classification project aims to analyze and classify messages to improve communication during disasters, using data provided by <a href="https://appen.com/">Appen</a> (formally Figure 8) that contains real messages that were sent during disaster events.

A web application can be hosted locally to classify new messages. 

The result should look like this:

<details>
<summary><b>Home</b></summary>
<img src="https://user-images.githubusercontent.com/84249222/161833153-c5185f3d-d61a-4c8b-8563-e043a5568e81.png">
</details>
<details>
<summary><b>Visualizing data (1)</b></summary>
<img src="https://user-images.githubusercontent.com/84249222/161833720-af76d38e-8113-446a-aec7-612535e6b57e.png">
</details>
<details>
<summary><b>Visualizing data (2)</b></summary>
<img src="https://user-images.githubusercontent.com/84249222/161833827-a724bf42-1c43-4fb4-9165-6584f07e9342.png">
</details>
<details>
<summary><b>Message Classification</b></summary>
<img src="https://user-images.githubusercontent.com/84249222/161833642-47163ee7-4c1f-4181-b0f9-282260ed4c32.png">
</details>
<details>
<summary><b>About the Project</b></summary>
<img src="https://user-images.githubusercontent.com/84249222/161834342-bfbf160e-b758-4dd4-a687-3fe55ea6d913.png">
</details>

# Directories

## Config
Configuration setup is handled in the **"config"** directory through two files: *core.py* and *config.yml*. The different path files needed to run this project (messages and categories data, database file, and model pickle file) are specified in this folder, and a validation is done to ensure everything works as intented. Thus users do not need to write additional name files when running the python script, which can prevent errors and increases efficiency.

To re-train the model, e.g. if new data becomes available, the *"config.yml"* should be updated with the new file names, or the previous data files should be replaced.

## Data
It contains the raw data (messages and categories) in csv files, as well as the cleaned data in a database (.db) file.

It also contains the python script needed to apply the entire ETL process, *process_data.py*, which extracts data from csv files, transforms them and then loads them into a single SQLite database.

To run this script on the command line, from the project folder: 
<br> `python data/process_data.py`

## Models
It contains the python script that handles all the machine learning steps needed for this project, *train_classifier.py*. It also holds the pickle file containing the best model from the GridSearchCV done on the training set.

To run the python script, *train_classifier.py*, from the command line:
<br> `python models/train_classifier.py`

<details>
<summary><b>Training</b></summary>
<img src="https://user-images.githubusercontent.com/84249222/161833476-7831840d-9ed7-4caf-89a2-e6043a7db055.png">
</details>


## App
It contains the necessary files to run the wep application. This include two python scripts:
* *run.py* which contains the Flask code needed to render the HTML files as well as the Plotly figures 
* *functions.py* which contains extra functions needed to execute *run.py* (for a modular and clean code)

In addition, two additional directories: templates and static, contain the necessary HTML and CSS files.

To access the web application on a local computer, run: `python app/run.py` and run the given url.