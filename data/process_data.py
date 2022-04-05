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
    """Loads and merge the messages & categories datasets"""
    # Loads datasets
    messages = pd.read_csv(Path(f'{DATASET_DIR/messages_file_name}'))
    categories = pd.read_csv(Path(f'{DATASET_DIR/categories_file_name}'))
    
    # Merges the two datasets
    data = messages.merge(categories, on='id')
    
    return data


def clean_data(data):
    """Loads data, split categories into multiple columns, 
    then remove duplicates"""
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


def save_data(df):
    """Saves cleaned data into an sqlite database"""
    save_file_name = config.app_config.database_file_name
    table_name = config.app_config.table_name
    
    engine = create_engine(f'sqlite:///{DATASET_DIR}/{save_file_name}')
    df.to_sql(f'{table_name}', engine, index=False, if_exists='replace')
    
    
def etl_process():
    """Contains the 3 steps of the ETL pipeline"""
    df = load_data(messages_file_name = config.app_config.messages_file_name,
                   categories_file_name = config.app_config.categories_file_name)  
    print('extracted data')
    df = clean_data(df)
    print('transformed data')
    save_data(df)
    print('loaded data')
    
    
if __name__ == '__main__':
    # run data pipeline
    etl_process()