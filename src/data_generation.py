from datetime import datetime
import json
from data import tickers_loader, target_feature
from features import technical_analysis
from models import train_test_split


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers = config['tickers']
    start_date = datetime.fromisoformat(config['start_date'])
    end_date = datetime.fromisoformat(config['end_date'])
    tickers_dir_path = config['data']['tickers_path']
    tickers_ta_path = config['data']['tickers_ta_path']
    tickers_train_path = config['data']['train_path']
    tickers_val_path = config['data']['validation_path']
    tickers_test_path = config['data']['test_path']
    target_var_path = config['data']['target_var_path']
    train_size = config['modelling']['split']['train_size']
    validation_size = config['modelling']['split']['validation_size']
    gap = config['modelling']['split']['gap']

    tickers_loader.load_and_store(tickers, start_date, end_date, tickers_dir_path)
    target_feature.generate_target_feature(tickers, tickers_dir_path, target_var_path)
    technical_analysis.perform_ta(tickers, tickers_dir_path, tickers_ta_path)

    for ticker in tickers:
        train_test_split.split(
            ticker,
            tickers_ta_path,
            tickers_train_path,
            tickers_val_path,
            tickers_test_path,
            train_size,
            validation_size,
            gap,
        )


if __name__ == '__main__':
    main()
