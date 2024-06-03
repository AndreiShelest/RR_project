from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
import json
from xgboost import XGBClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from pathlib import Path
from system_types import create_pipeline, system_types
from constants import date_index_label, signal_label
import pywt
import numpy as np
from skimage.restoration import denoise_wavelet
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint

'''
CURRENTLY NOT USED
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
'''

class Wavelet(BaseEstimator, TransformerMixin):

    def __init__(self, mode, level, wavelet) -> None:
        self.wavelet = wavelet
        self.mode = mode
        self.level = level

    def fit(self, X, y=None):
        # Calculate the thresholds from the training data
        self.train_data_ = X.copy()

        self.thresholds_ = []

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            # each feature is done separetely
            feature_data = X[:, feature_idx]
            coeffs = pywt.wavedec(feature_data, self.wavelet, level=self.level)

            threshold = self.threshold_coefficients(coeffs)
            self.thresholds_.append(threshold)
        return self

    def threshold_coefficients(self, coeffs):
        # applies thresholding optimisation
        denoised_coeffs = [coeffs[0]]  
        denoised_coeffs += [
            denoise_wavelet(
                c,
                method='BayesShrink',
                mode=self.mode,
                wavelet_levels=self.level,
                wavelet=self.wavelet,
                rescale_sigma=True,
            )
            for c in coeffs[1:]
        ]
        return denoised_coeffs

    def transform(self, X, y=None):
        # incremental denoising
        n_samples, n_features = X.shape
        X_denoised = np.zeros_like(X)

        for feature_idx in range(n_features):
            feature_data = X[:, feature_idx]
            train_feature_data = self.train_data_[:, feature_idx]
            denoised_feature_data = np.zeros(n_samples)
            extended_data = np.concatenate([train_feature_data, feature_data])

            for i in range(n_samples):
                # data_point = extended_data[len(train_feature_data) + i]
                data_up_to_point = extended_data[: len(train_feature_data) + i + 1]

                coeffs = pywt.wavedec(data_up_to_point, self.wavelet, level=self.level)
                coeffs_thresholded = self.threshold_coefficients(coeffs)
                denoised_signal = self.reconstruct_signal(coeffs_thresholded)

                denoised_point = denoised_signal[-1]
                denoised_feature_data[i] = denoised_point

            X_denoised[:, feature_idx] = denoised_feature_data

        print("Wavelet Transformed Data Shape:", X_denoised.shape)

        return X_denoised

    def reconstruct_signal(self, coeffs):
        return pywt.waverec(coeffs, self.wavelet)


def _generate_test_signal(
    ticker,
    system_type,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val:pd.DataFrame,
    Y_val: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    pca_settings,
    dwt_params,
    xgboost,
    seed,
):  
    param_search = {
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.3),
            "min_child_weight": [],
            "subsample": [0.9],
            "early_stopping_rounds": [10]
        }
    model = create_pipeline(
        system_type,
        normalizer=MinMaxScaler(),
        pca=PCA(**pca_settings),
        dwt=Wavelet(**dwt_params),
        xgboost=XGBClassifier(**xgboost, seed=seed),
    )

    grid_search = GridSearchCV(pipeline, param_search, cv=5, n_jobs=-1, scoring='accuracy')
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
    tickers_test_path,
    target_feature_path,
    signal_path,
    pca_settings,
    dwt_params,
    xgboost,
    seed,
):
    for ticker in tickers:
        train_df = pd.read_csv(
            f'{tickers_train_path}/{ticker}.csv', index_col=date_index_label
        )
        train_df.dropna(inplace=True)

        train_feat_df = pd.read_csv(
            f'{target_feature_path}/{ticker}.csv', index_col=date_index_label
        ).loc[train_df.index]

        test_df = pd.read_csv(
            f'{tickers_test_path}/{ticker}.csv', index_col=date_index_label
        )
        test_feat_df = pd.read_csv(
            f'{target_feature_path}/{ticker}.csv', index_col=date_index_label
        ).loc[test_df.index]

        for system_type in system_types:
            signal_df = _generate_test_signal(
                ticker,
                system_type,
                train_df,
                train_feat_df,
                test_df,
                test_feat_df,
                pca_settings,
                dwt_params,
                xgboost_settings,
                seed,
            )

            Path(f'{signal_path}/{system_type}').mkdir(exist_ok=True)
            signal_df.to_csv(f'{signal_path}/{system_type}/{ticker}.csv')


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers_train_path = config['data']['train_path']
    tickers_test_path = config['data']['test_path']
    target_feature_path = config['data']['target_var_path']
    tickers = config['tickers']
    signal_path = config['data']['signal_path']
    pca_settings = config['modelling']['pca']
    dwt_params = config['modelling']['dwt']
    xgboost_settings = config['modelling']['xgboost']
    seed = config['seed']

    perform_modelling(
        tickers,
        tickers_train_path,
        tickers_test_path,
        target_feature_path,
        signal_path,
        pca_settings,
        dwt_params,
        xgboost_settings,
        seed,
    )


if __name__ == '__main__':
    main()