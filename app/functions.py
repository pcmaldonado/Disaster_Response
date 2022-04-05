# ===== LIBRARIES ======
# To handle data
import pandas as pd
from sqlalchemy import create_engine

# To access configuration
import sys; sys.path.append('.')
from config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

# To import model
import joblib

# To create visualizations
import plotly.graph_objs as go


# ===== FUNCTIONS ======
def load_clean_data():
    """Loads clean data from SQLite database to use data for visualizations
    
    Arguments: 
        None
    Returns:
        features, targets: pandas dataframes 
    """
    # To load data from SQLite database
    database_file_name = config.app_config.database_file_name
    table_name = config.app_config.table_name
    engine = create_engine(f'sqlite:///{DATASET_DIR}\\{database_file_name}')
    df = pd.read_sql_table(f'{table_name}', engine)

    # Separate features from targets
    features = df[config.model_config.features]
    targets = df[config.model_config.targets]

    return features, targets


def load_model():
    """Loads the fitted model
    
    Arguments:
        None
    Returns:
        trained_model: trained sklearn model
    """
    file_name = f'{config.app_config.model_save_file}'
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename = file_path)
    
    return trained_model  


def return_figures():
    """Creates and returns Plotly visualizations
    
    Arguments:
        None
    Returns:
        figures: list of Plotly figures to display on web app
    """
    # Convert the plotly figures to JSON for javascript in html template
    features, targets = load_clean_data()
    cat_names = targets.mean().sort_values(ascending=False).index
    cat_perc = targets.mean().sort_values(ascending=False).values * 100


    # Figure1 - category percentages
    graph_one = [go.Bar(
                        x = cat_names,
                        y = cat_perc,
                        marker_color='rgb(158,202,225)'
                )]

    layout_one = dict(title = 'Percentage of messages corresponding to each category',
                        xaxis = dict(automargin=True,
                                    title='Categories', 
                                    standoff=60, 
                                    tickangle=320,
                                    # size=30, 
                                    # color='crimson',
                                    ),

                        yaxis = dict(title = 'Percentage'),
                        margin=dict(
                            b=200,
                            pad=4,
                        )
                        
                        )

    # Figure2
    targets = targets.drop(['child_alone'], axis = 1)
    corr = targets.corr(method='spearman')
    graph_two = [go.Heatmap(
                        z = corr.values,
                        x = corr.index.values,
                        y = corr.columns.values,
                        colorscale='Viridis'
                )]

    layout_two = dict(title = 'Spearman correlation between categories',
                        height=800,
                        margin=dict(
                                    l=200,
                                    r=200,
                                    b=100,
                                    t=50,
                                    pad=4
                                ),
                        xaxis = dict(automargin=True,
                                    standoff=60, 
                                    tickangle=320,
                                    ),

                        # yaxis = dict(title = 'Percentage'),
                        
                        )

    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))

    return figures