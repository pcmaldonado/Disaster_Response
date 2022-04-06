# ===== LIBRARIES ======
# Basic libraries to handle data
import numpy as np
import pandas as pd

# To load data to SQL database
from sqlalchemy import create_engine

# To access configuration variables
import sys; sys.path.append('.')
from pathlib import Path
from config.core import DATASET_DIR, config 


# ===== FUNCTIONS ======
def load_data(messages_file_name:str, categories_file_name:str):
    """Loads and merge the messages & categories datasets
    
    Arguments:
        messages_file_name: name file of message data
        categories_file_name: name file of categories data
    Returns:
        data: pandas dataframe
    """
    # Loads datasets
    messages = pd.read_csv(Path(f'{DATASET_DIR/messages_file_name}'))
    categories = pd.read_csv(Path(f'{DATASET_DIR/categories_file_name}'))
    
    # Merges the two datasets
    data = messages.merge(categories, on='id')
    
    return data


def clean_data(data):
    """Loads data, splits categories into multiple columns, 
    then removes duplicates
    
    Arguments: 
        data: pandas dataframe
    Returns:
        data: pandas dataframe
    """
    # ===== Transforming "Categories" ======
    # Splitting categories (one col) into n columns
    cats = data.categories.str.split(';', expand=True)
    cols = cats.iloc[0,:].apply(lambda x:x[:-2])
    cats.columns = cols  
    for column in cats:
        cats[column] = pd.to_numeric(cats[column].str[-1])
        
    # Concat categories with the rest of the data
    data = pd.concat([data.drop(['categories'], axis=1), cats], axis=1)
    
    # ===== Removing duplicates ======
    # Removing unnecesary feature
    data = data.drop_duplicates()
    
    return data


def save_data(data):
    """Saves cleaned data into an SQLite database following 
    configuration parameters for path file
    
    Arguments:
        data: pandas dataframe
    Returns:
        None
    """
    save_file_name = config.app_config.database_file_name
    table_name = config.app_config.table_name
    
    engine = create_engine(f'sqlite:///{DATASET_DIR}\{save_file_name}')
    data.to_sql(f'{table_name}', engine, index=False, if_exists='replace')

    print(f'sqlite:///{DATASET_DIR}\{save_file_name}')
    

def etl_process():
    """Contains the 3 steps of the ETL pipeline
    
    Arguments:
        None
    Returns:
        None
    """
    data = load_data(messages_file_name = config.app_config.messages_file_name,
                   categories_file_name = config.app_config.categories_file_name)  
    print('Data extraction complete')

    data = clean_data(data)
    print('Data transformed correctly')

    save_data(data)
    print('Loaded clean data into SQLite database')
    
    
if __name__ == '__main__':
    # run data pipeline
    etl_process()