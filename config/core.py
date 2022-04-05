# ==== LIBRARIES ====
# import sys; sys.path.append('.')
from pathlib import Path
from pydantic import BaseModel
from typing import List
from strictyaml import YAML, load


# ==== DIRECTORIES ====
PACKAGE_ROOT = Path().resolve()
CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_FILE_PATH = CONFIG_DIR / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "models"

# ==== CONFIGURATIONS ====
class AppConfig(BaseModel):
    """Application-level config"""

    package_name: str
    messages_file_name: str
    categories_file_name: str
    database_file_name: str
    table_name: str
    model_save_file: str
    

class ModelConfig(BaseModel):
    """All configuration relevant to model training"""
    
    targets: List[str]
    features: str
    

class Config(BaseModel):
    """Master config object"""
    app_config: AppConfig
    model_config: ModelConfig
    

def find_config_file(): 
    """Locate the configuration file"""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH}")


def fetch_config_from_yaml(cfg_path: Path = None):
    """Parse YAML containing the package configuration"""
    if not cfg_path:
        cfg_path = find_config_file()
    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None):
    """Run validation on config values"""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
    
    # specify the data distribution from the strityaml YAML type
    _config = Config(
        app_config = AppConfig(**parsed_config.data),
        model_config = ModelConfig(**parsed_config.data)
    )

    return _config

config = create_and_validate_config()

