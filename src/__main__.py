from datetime import datetime
import json
from data import tickers_loader
import os

def main():
    print(os.getcwd())
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers = config['tickers']
    start_date = datetime.fromisoformat(config['start_date'])
    end_date = datetime.fromisoformat(config['end_date'])
    tickers_dir_path = config['data']['tickers_path']

    loaded_tickers = tickers_loader.load_and_store(tickers,
                                                   start_date,
                                                   end_date,
                                                   tickers_dir_path)

main()