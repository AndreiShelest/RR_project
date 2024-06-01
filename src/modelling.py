from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
import json
from xgboost import XGBClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from pathlib import Path
from system_types import create_pipeline, system_types


class Debug(BaseEstimator, TransformerMixin):

    def __init__(self, ticker, df_index, interim_path) -> None:
        super().__init__()
        self.ticker = ticker
        self.df_index = df_index
        self.interim_path = interim_path

    def transform(self, X):
        return X

    def fit(self, X, y=None, **fit_params):
        transformed_df = pd.DataFrame(
            data=X,
            columns=[f'C{idx}' for idx, _ in enumerate(X[0])],
        )
        transformed_df.insert(0, self.df_index.name, self.df_index)
        transformed_df.set_index(self.df_index.name, inplace=True)

        transformed_df.to_csv(f'{self.interim_path}/{self.ticker}.csv')
        return self


def _generate_test_signal(
    config,
    ticker,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
):
    signal_path = config['data']['signal_path']
    pca_components = config['modelling']['pca']['components']
    # interim_train_path = config['data']['interim_train_path']

    xgboost_settings = config['modelling']['xgboost']

    for system_type in system_types:
        model = create_pipeline(
            system_type,
            normalizer=MinMaxScaler(),
            pca=PCA(n_components=pca_components),
            xgboost=XGBClassifier(**xgboost_settings),
        )
        model.fit(X_train, Y_train)

        test_score = model.score(X_test, Y_test)
        print(f'System={system_type}, ticker={ticker}, test_score={test_score}')

        signal = model.predict(X_test)

        signal_df = pd.DataFrame({'Signal': signal}, index=X_test.index)

        Path(f'{signal_path}/{system_type}').mkdir(exist_ok=True)
        signal_df.to_csv(f'{signal_path}/{system_type}/{ticker}.csv')


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers_train_path = config['data']['train_path']
    tickers_test_path = config['data']['test_path']

    target_feature_path = config['data']['target_var_path']
    tickers = config['tickers']

    for ticker in tickers:
        train_df = pd.read_csv(f'{tickers_train_path}/{ticker}.csv', index_col='Date')
        train_df.dropna(inplace=True)

        train_feat_df = pd.read_csv(
            f'{target_feature_path}/{ticker}.csv', index_col='Date'
        ).loc[train_df.index]

        test_df = pd.read_csv(f'{tickers_test_path}/{ticker}.csv', index_col='Date')
        test_feat_df = pd.read_csv(
            f'{target_feature_path}/{ticker}.csv', index_col='Date'
        ).loc[test_df.index]

        _generate_test_signal(
            config, ticker, train_df, train_feat_df, test_df, test_feat_df
        )


if __name__ == '__main__':
    main()
