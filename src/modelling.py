from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
import json
from xgboost import XGBClassifier
from pathlib import Path
from system_types import create_pipeline, system_types
from constants import date_index_label, signal_label
from custom_estimators import Wavelet


def _generate_test_signal(
    ticker,
    system_type,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    pca_settings,
    dwt_params,
    xgboost_settings,
    strategy_settings,
    seed,
):

    model = create_pipeline(
        system_type,
        normalizer=MinMaxScaler(),
        pca=PCA(**pca_settings),
        dwt=Wavelet(**dwt_params),
        xgboost=XGBClassifier(**xgboost_settings, seed=seed),
    )
    model.fit(X_train, Y_train)

    if model.named_steps.get('pca') is not None:
        print(
            f'System={system_type}, ticker={ticker}, PCA components={model["pca"].n_components_}'
        )

    test_score = model.score(X_test, Y_test)
    print(f'System={system_type}, ticker={ticker}, test_score={test_score}')

    signal = model.predict(X_test)

    signal_df = pd.DataFrame({signal_label: signal}, index=X_test.index)
    return signal_df


def perform_modelling(
    tickers,
    tickers_train_path,
    tickers_validation_path,
    tickers_test_path,
    target_feature_path,
    signal_path,
    pca_settings,
    dwt_params,
    xgboost_settings,
    strategy_settings,
    seed,
):
    def get_features(ticker, path):
        return pd.read_csv(f'{path}/{ticker}.csv', index_col=date_index_label)

    def get_labels(ticker, feat_index):
        return pd.read_csv(
            f'{target_feature_path}/{ticker}.csv', index_col=date_index_label
        ).loc[feat_index]

    for ticker in tickers:
        train_df = get_features(ticker, tickers_train_path)
        train_df.dropna(inplace=True)
        train_feat_df = get_labels(ticker, train_df.index)

        valid_df = get_features(ticker, tickers_validation_path)
        valid_feat_df = get_labels(ticker, valid_df.index)

        test_df = get_features(ticker, tickers_test_path)
        test_feat_df = get_labels(ticker, test_df.index)

        for system_type in system_types:
            signal_df = _generate_test_signal(
                ticker,
                system_type,
                train_df,
                train_feat_df,
                valid_df,
                valid_feat_df,
                test_df,
                test_feat_df,
                pca_settings,
                dwt_params,
                xgboost_settings,
                strategy_settings,
                seed,
            )

            Path(f'{signal_path}/{system_type}').mkdir(exist_ok=True)
            signal_df.to_csv(f'{signal_path}/{system_type}/{ticker}.csv')


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers_train_path = config['data']['train_path']
    tickers_validation_path = config['data']['validation_path']
    tickers_test_path = config['data']['test_path']
    target_feature_path = config['data']['target_var_path']
    tickers = config['tickers']
    signal_path = config['data']['signal_path']
    pca_settings = config['modelling']['pca']
    dwt_params = config['modelling']['dwt']
    xgboost_settings = config['modelling']['xgboost']
    strategy_settings = config['strategy']
    seed = config['seed']

    perform_modelling(
        tickers,
        tickers_train_path,
        tickers_validation_path,
        tickers_test_path,
        target_feature_path,
        signal_path,
        pca_settings,
        dwt_params,
        xgboost_settings,
        strategy_settings,
        seed,
    )


if __name__ == '__main__':
    main()
