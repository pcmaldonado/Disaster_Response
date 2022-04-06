# ===== LIBRARIES ======
# To handle data
import pandas as pd
from sqlalchemy import create_engine

# To build ML pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score

# To get execution time 
import time

# To save model
import joblib

# To preprocess text data
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import re

# To access configuration 
import sys; sys.path.append('.')
from config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


# ===== FUNCTIONS ======
def prepare_data():
    """It loads the data from the SQLite database,
    separate targets and features, then splits train/test sets 
    and returns X_train, X_test, y_train, y_test
    
    Arguments:
        None
    Returns:
        X_train, X_test, y_train, y_test: pandas dataframes containing training and test data
    """
    # load data from database
    database_file_name = config.app_config.database_file_name
    table_name = config.app_config.table_name
    engine = create_engine(f'sqlite:///{DATASET_DIR}/{database_file_name}')
    df = pd.read_sql_table(f'{table_name}', engine)

    # Separate features from targets
    X = df[config.model_config.features]
    y = df[config.model_config.targets]
    
    # Splits data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Data ready for modeling')
    return X_train, X_test, y_train, y_test


def tokenize(text):
    '''Normalize and tokenize input text,
    then applies lemmatization, finally returns cleaned text
    
    Arguments:
        text: 'str' input text to be transformed
    Returns:
        text: 'str' cleaned text ready for modeling
    '''
    
    # Makes all text lowercase and removes whitespace
    # Then keeps only alphabetical characters
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z]',' ',text)
    
    # Tokenize text 
    text = word_tokenize(text)
    
    # Lemmatize text
    text = [WordNetLemmatizer().lemmatize(t) for t in text]
    
    return text


def build_model():
    # ML pipeline
    """It builds the pipeline and sets the GridSearchCV parameters
    
    Arguments:
        None
    Returns:
        model_pipeline: sklearn pipeline containing NLP pipeline,
                        RandomForestClassifier and GridSearchCV
    """
    model_pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
                    ])
     
    param_grid = {
        'vect__max_df': (0.65, 0.75), #Ignore terms that have a frequency higher than the given threshold
        'tfidf__use_idf': (True, False), #Enable inverse-document-frequency reweighting
        'clf__min_samples_split': [2, 3], #The minimum number of samples required to split an internal node
    }
    
    scorer = make_scorer(f1_score, average='weighted', zero_division=0)
    model_pipeline = GridSearchCV(model_pipeline, param_grid=param_grid, scoring=scorer, cv=3, verbose=2)
        
    return model_pipeline


def train_model(X_train, y_train):
    """Trains the model on the train set
    
    Arguments:
        X_train, y_train: pandas dataframe with training data
    Returns:
        model: sklearn model with best hyperparameters
    """
   # Training Model
    print('Training the ML model')
    model = build_model()

    startTime = time.time()
    model.fit(X_train, y_train)
    print('Done training')
    
    # Getting best hyperparameteres from GridSearchCV
    executionTime = (time.time() - startTime)
    print('Execution time in minutes: ' + str(executionTime/60))
    print('Best hyperpameters:', model.best_params_)
    
    return model.best_estimator_
    

def evaluate_model(model, X_test, y_test):
    """Predicts on test data then shows performance metrics 
    (weighted f1-score, recall & precision)
    
    Arguments:
        model: sklearn model to use for prediction
        X_test, y_test: pandas dataframe with test data
    Returns:
        None
    """
    # Predicts on test data
    y_pred = model.predict(X_test)
    target_names = y_test.columns
    
    # Prints performance on test data by category
    print('\n\nPerformance by category:')
    print('-------------------------')
    for i in range(len(target_names)):
        f1score = f1_score(y_test.iloc[:, i].values, y_pred[:, i], average='weighted', zero_division=0)
        recall = recall_score(y_test.iloc[:, i].values, y_pred[:, i], average='weighted', zero_division=0)
        precision = precision_score(y_test.iloc[:, i].values, y_pred[:, i], average='weighted', zero_division=0)
        print(f'Category: {target_names[i]}')
        print(f'F1-score: {f1score}')
        print(f'Recall: {recall}')
        print(f'Precision: {precision}\n\n')


def export_model(model):
    """Saves model as pickle file following configuration set path
    (see config/core.py and config/config.yml)
    
    Arguments:
        model: sklearn model to save
    Returns:
        None
    """
    # Saving model
    save_file_name = f'{config.app_config.model_save_file}'
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(value=model, filename=save_path, compress=3)
    print('Model saved')


def run_pipeline():
    """Runs the entire ML pipeline: data preparation, 
        model training, model validation and model saving

    Arguments:
        None
    Returns:
        None
    """
    X_train, X_test, y_train, y_test = prepare_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    export_model(model)
    

if __name__ == '__main__':
    # run ML pipeline
    run_pipeline()