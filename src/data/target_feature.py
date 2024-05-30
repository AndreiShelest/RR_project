import json
import pandas as pd
import numpy as np

def generate_target_feature(tickers, tickers_dir_path, target_var_path):
    target_dfs = []
    for ticker in tickers:
        ticker_df = pd.read_csv(f'{tickers_dir_path}/{ticker}.csv', index_col='Date')

        shifted_close = np.pad(ticker_df['Close'], pad_width=(0, 1), constant_values=(np.NaN, np.NaN))[1:]
        delta_close = (shifted_close - ticker_df['Close']) / ticker_df['Close']

        # the last one will always be False, as np.NaN is not comparable to 0
        buy_or_sell = delta_close >= 0 

        buy_or_sell_df = pd.DataFrame({'BuySell': buy_or_sell * 1}, index=ticker_df.index)
        buy_or_sell_df.to_csv(f'{target_var_path}/{ticker}.csv')

        target_dfs.append(buy_or_sell_df)

        print(f'BuySell for {ticker} is stored.')

    return target_dfs

def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers_dir_path = config['data']['tickers_path']
    target_var_path = config['data']['target_var_path']
    tickers = config['tickers']

    generate_target_feature(tickers, tickers_dir_path, target_var_path)

if __name__ == '__main__':
    main()