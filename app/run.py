# ===== LIBRARIES ======
# To create web app
from flask import Flask, render_template, request

# To create visualizations
import plotly
import json

# To access configuration 
import sys; sys.path.append('.')
# from models.train_classifier import nlp_pipeline #needed to load model without errors

# Functions needed to run this script
from models.functions import load_model, return_figures, load_clean_data

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# to clean input text
import re
def nlp_pipeline(text):
    '''Normalize and tokenize input text,
    then applies stemming and lemmatization,
    finally returns cleaned text'''
    
    # Makes all text lowercase then keeps only alphabetical characters
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]',' ',text)
    
    # Tokenize text 
    text = word_tokenize(text)
    
    # Stopwords
    omit = ['no', 'but']
    text = [t for t in text if t not in set(stopwords.words('english')) - set(omit)]
    
    # Stemming text
    text = [PorterStemmer().stem(t) for t in text]

    # Lemmatize text
    text = [WordNetLemmatizer().lemmatize(t) for t in text]
    
    return text

    
# ===== GETTING THE DATA ======
# load data
features, targets = load_clean_data()

# load model
model = load_model()




# ===== WEB APPLICATION ======
app = Flask(__name__, template_folder="./templates")

@app.route('/', methods = ["GET", "POST"])
@app.route('/index', methods = ["GET", "POST"])

def index():
    # Visualizations
    figures = return_figures()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]
    
    # Convert the plotly figures to JSON for javascript in html template
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(targets, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
                        'go.html',
                        query=query,
                        classification_result=classification_results
                    )

@app.route('/about')
def about():
    return render_template('about.html')

def main():
    app.run()

if __name__ == '__main__':
    main()
