from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import json
import numpy as np
from features import imputation

def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers_train_path = config['data']['train_path']
    interim_train_path = config['data']['interim_train_path']
    tickers = config['tickers']

    pipelines = {}

    for ticker in tickers:
        train_df = pd.read_csv(f'{tickers_train_path}/{ticker}.csv', index_col='date')

        ind_imputer = imputation.IndicatorImputer()

        pipelines[ticker] = Pipeline(
            [
                ('imputer', ind_imputer),
                ('normalizer', MinMaxScaler()),
            ]
        )

        transformed = pipelines[ticker].fit_transform(train_df)
        transformed_df = pd.DataFrame(data=transformed, columns=train_df.columns)
        transformed_df.insert(0, train_df.index.name, train_df.index)
        transformed_df.set_index(train_df.index.name, inplace=True)

        transformed_df.to_csv(f'{interim_train_path}/{ticker}.csv')


if __name__ == '__main__':
    main()