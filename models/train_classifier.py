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

# To access configuration 
import sys; sys.path.append('.')
from config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from models.functions import nlp_pipeline

# ===== FUNCTIONS ======
def prepare_data():
    """It loads the data from the SQL database,
    separate targets and features, then splits train/test sets 
    and returns X_train, X_test, y_train, y_test"""
    # load data from database
    database_file_name = config.app_config.database_file_name
    table_name = config.app_config.table_name
    engine = create_engine(f'sqlite:///{DATASET_DIR}\\{database_file_name}')
    df = pd.read_sql(f'SELECT * FROM {table_name}', engine)

    # Separate features from targets
    X = df[config.model_config.features]
    y = df[config.model_config.targets]
    
    # Simplification to avoid multiclass problems (plus, class 2 is highly imbalanced)
    y['related'] = y['related'].replace({2:1})
    
    # Splits data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Data ready for modeling')
    return X_train, X_test, y_train, y_test



def build_model():
    # ML pipeline
    """It builds the pipeline and sets the GridSearchCV parameters"""
    model_pipeline = Pipeline([('vect', CountVectorizer(tokenizer=nlp_pipeline)), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
                    ])
     
    param_grid = {
        'vect__max_df': (0.7, 0.8), #Ignore terms that have a frequency higher than the given threshold
        'tfidf__use_idf': (True, False), #Enable inverse-document-frequency reweighting
        'clf__max_features': ['sqrt', 'log2'], #Number of features to consider when looking for the best split
    }
    
    scorer = make_scorer(f1_score, average='weighted', zero_division=0)
    model_pipeline = GridSearchCV(model_pipeline, param_grid=param_grid, scoring=scorer, cv=3, verbose=2)
        
    return model_pipeline


def train_model(X_train, y_train):
    """Trains the model on the train set"""
   # Training Model
    model = build_model()

    print('Training the ML model')
    startTime = time.time()
    model.fit(X_train, y_train)

    # Getting best hyperparameteres from GridSearchCV
    print('Done training')

    executionTime = (time.time() - startTime)
    print('Execution time in minutes: ' + str(executionTime/60))
    print('Best hyperpameters:', model.best_params_)
    
    return model
    

def evaluate_model(model, X_test, y_test):
    """Predicts on test data then shows performance metrics (weighted f1-score)"""
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
    """Saves model as pickle file"""
    # Saving model
    save_file_name = f'{config.app_config.model_save_file}'
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(value=model, filename=save_path, compress=3)
    print('Model saved')


def run_pipeline():
    """Runs the entire ML pipeline"""
    X_train, X_test, y_train, y_test = prepare_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    export_model(model)
    

if __name__ == '__main__':
    # run ML pipeline
    run_pipeline()