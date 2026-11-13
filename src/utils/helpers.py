"""
Shared helpers: the Meal record, pickle I/O, file loggers, cached downloads, and the loaders for
the processed meal tables and the food-nutrient table. src/preprocessing/data_preprocessing.py
uses Meal, to_pickle, download_file, fetch_data and setup_logger; cli/run_clustering.py uses
setup_logger, load_meal_data and load_nutrient_data; cli/evaluate.py, cli/run_multi_meal_cvae.py
and the experiments/ scripts use setup_logger.
"""
import os
import pickle
import requests
import pandas as pd

import logging

class Meal:
    def __init__(self, id, food_codes, gram_portions, day_eaten, food_source):
        self.id = id
        self.food_codes = food_codes
        self.gram_portions = gram_portions
        self.day_eaten = day_eaten
        self.food_source = food_source

def to_pickle(data, filename):
    # Ensure the parent directory exists before saving
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
def setup_logger(caller_name, log_file, level=logging.INFO, file_mode: str = "w"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file, mode=file_mode)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(caller_name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger

def download_file(url, local_filepath, log):
    try:
        # Ensure the parent directory exists before downloading
        os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        if log is not None:
            log.error(f"Error downloading file {url}: {e}")
           
def fetch_data(file_url, cache_dir, log):
    try:
        filename = os.path.basename(file_url)
        local_filepath = os.path.join(cache_dir, filename)
        
        if not os.path.exists(local_filepath):
            if log is not None:
                log.info(f"Downloading {filename}...")
            download_file(file_url, local_filepath, log)
        else:
            if log is not None:
                log.info(f"Using cached {filename}")
        
        return local_filepath
    except Exception as e:
        if log is not None:
            log.error(f"Error fetching data {file_url}: {e}")
        raise

def load_meal_data():
    """Per-meal-type grams tables (rows = meals, columns = food codes) from
    data/processed/real_meals/dataframes/<meal>_bci_filtered.pkl (written by `make preprocess`)."""
    meal_data = {}
    for meal_type in ['breakfast', 'lunch', 'dinner']:
        bci_file = f'data/processed/real_meals/dataframes/{meal_type}_bci_filtered.pkl'
        if not os.path.exists(bci_file):
            raise FileNotFoundError(f"{bci_file} not found; run `make preprocess` first.")
        meal_data[meal_type] = pd.read_pickle(bci_file)
        print(f"Loaded {len(meal_data[meal_type])} {meal_type} meals (bci_filtered)")
    return meal_data

def load_nutrient_data():
    """data/mappings/food_to_nutrient.pkl indexed by 'Food code' (nutrients per 100 g);
    non-numeric values become 0."""
    nutrient_file = 'data/mappings/food_to_nutrient.pkl'
    if not os.path.exists(nutrient_file):
        raise FileNotFoundError(f"Error: {nutrient_file} not found. Please run data preprocessing first.")
    
    nutrient_data = pd.read_pickle(nutrient_file)
    nutrient_data.set_index('Food code', inplace=True)
    nutrient_data = nutrient_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    print(f"Loaded nutrient data for {len(nutrient_data)} foods")
    
    return nutrient_data