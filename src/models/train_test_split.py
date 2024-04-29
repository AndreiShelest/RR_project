import json
import pandas as pd
import numpy as np

def split(ticker: str,
          tickers_ta_path: str,
          tickers_train_path: str,
          tickers_val_path: str,
          tickers_test_path: str,
          train_size: float,
          validation_size: float):
    ticker_df = pd.read_csv(f'{tickers_ta_path}/{ticker}.csv', index_col='date')
    df_size = len(ticker_df)

    df_train_size = np.floor(train_size * df_size).astype(int)
    df_val_size = np.floor(validation_size * df_size).astype(int)

    train_df = ticker_df[:df_train_size]
    val_df = ticker_df[df_train_size:df_train_size + df_val_size]
    test_df = ticker_df[df_train_size + df_val_size:]

    train_df.to_csv(f'{tickers_train_path}/{ticker}.csv')
    val_df.to_csv(f'{tickers_val_path}/{ticker}.csv')
    test_df.to_csv(f'{tickers_test_path}/{ticker}.csv')

    print(f'Train-Val-Test split for ticker {ticker} is completed.')


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers_ta_path = config['data']['tickers_ta_path']
    tickers_train_path = config['data']['train_path']
    tickers_val_path = config['data']['validation_path']
    tickers_test_path = config['data']['test_path']
    tickers = config['tickers']

    train_size = config['modelling']['split']['train_size']
    validation_size = config['modelling']['split']['validation_size']

    for ticker in tickers:
       split(ticker,
             tickers_ta_path,
             tickers_train_path,
             tickers_val_path,
             tickers_test_path,
             train_size,
             validation_size)

if __name__ == '__main__':
    main()