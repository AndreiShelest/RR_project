from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import pandas as pd
import json
from xgboost import XGBClassifier
from sklearn.base import TransformerMixin, BaseEstimator


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


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers_train_path = config['data']['train_path']
    tickers_test_path = config['data']['test_path']
    interim_train_path = config['data']['interim_train_path']
    signal_path = config['data']['signal_path']
    target_feature_path = config['data']['target_var_path']
    tickers = config['tickers']

    pca_components = config['modelling']['pca']['components']

    pipelines = {}

    for ticker in tickers:
        train_df = pd.read_csv(f'{tickers_train_path}/{ticker}.csv', index_col='Date')
        train_df.dropna(inplace=True)

        train_feat_df = pd.read_csv(
            f'{target_feature_path}/{ticker}.csv', index_col='Date'
        ).loc[train_df.index]

        pca = PCA(n_components=pca_components)
        bst = XGBClassifier(
            booster='gbtree',
            n_estimators=100,
            objective='binary:logistic',
            eval_metric='auc',
        )

        pipeline_steps = [
            ('normalizer', MinMaxScaler()),
            ('pca', pca),
            ('debug', Debug(ticker, train_df.index, interim_train_path)),
            ('xgboost', bst),
        ]

        pipelines[ticker] = Pipeline(pipeline_steps)
        pipelines[ticker].fit(train_df, train_feat_df)

        test_df = pd.read_csv(f'{tickers_test_path}/{ticker}.csv', index_col='Date')
        test_feat_df = pd.read_csv(
            f'{target_feature_path}/{ticker}.csv', index_col='Date'
        ).loc[test_df.index]

        test_score = pipelines[ticker].score(test_df, test_feat_df)
        print(f'{ticker}; test_score={test_score}')

        signal = pipelines[ticker].predict(test_df)
        signal_df = pd.DataFrame({'Signal': signal}, index=test_df.index)
        signal_df.to_csv(f'{signal_path}/{ticker}.csv')


if __name__ == '__main__':
    main()
