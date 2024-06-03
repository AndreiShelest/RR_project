from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import pandas as pd
import json
from xgboost import XGBClassifier
from sklearn.base import TransformerMixin, BaseEstimator
import pywt
import numpy as np
from skimage.restoration import denoise_wavelet


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
    
class Wavelet(BaseEstimator, TransformerMixin):

    def __init__(self, ticker, df_index, interim_path, mode, level, wavelet) -> None:
        self.ticker = ticker
        self.df_index = df_index
        self.interim_path = interim_path
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
        return denoise_wavelet(coeffs, method='BayesShrink', mode=self.mode, wavelet_levels=len(coeffs)-1, wavelet=coeffs[0].wavelet.name, rescale_sigma=True)

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
                data_point = extended_data[len(train_feature_data) + i]
                data_up_to_point = extended_data[:len(train_feature_data) + i + 1]

                coeffs = pywt.wavedec(data_up_to_point, self.wavelet, level=self.level)
                coeffs_thresholded = self.threshold_coefficients(coeffs)
                denoised_signal = self.reconstruct_signal(coeffs_thresholded)

                denoised_point = denoised_signal[-1]
                denoised_feature_data[i] = denoised_point

            X_denoised[:, feature_idx] = denoised_feature_data

        return X_denoised

    
    def reconstruct_signal(self, coeffs):
        return pywt.waverec(coeffs, self.wavelet)

    

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

    dwt_param = config['modelling']['dwt']
    mode=dwt_param["mode"]
    level=dwt_param['decomposition_level']
    wavelet=dwt_param['wavelet']


    pipelines = {}

    for ticker in tickers:
        train_df = pd.read_csv(f'{tickers_train_path}/{ticker}.csv', index_col='Date')
        train_df.dropna(inplace=True)

        train_feat_df = pd.read_csv(
            f'{target_feature_path}/{ticker}.csv', index_col='Date'
        ).loc[train_df.index]

        
        pca = PCA(n_components=pca_components)
        dwt = Wavelet(mode, level, wavelet)
        bst = XGBClassifier(
            booster='gbtree',
            n_estimators=100,
            objective='binary:logistic',
            eval_metric='auc',
        )


        pipeline_steps = [
            ('normalizer', MinMaxScaler()),
            ('pca', pca),
            ('dwt', dwt),
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
