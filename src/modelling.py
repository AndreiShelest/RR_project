from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import json
import numpy as np
from features import imputation

def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers_train_path = config['data']['train_path']
    tickers = config['tickers']

    for ticker in tickers:
        train_df = pd.read_csv(f'{tickers_train_path}/{ticker}.csv', index_col='date')

        ind_imputer = imputation.IndicatorImputer()
        ind_imputer.fit(train_df)
        imputed_df = ind_imputer.transform(train_df)


if __name__ == '__main__':
    main()