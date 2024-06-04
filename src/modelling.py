from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.pipeline import Pipeline
import json
from xgboost import XGBClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from pathlib import Path
from system_types import create_pipeline, system_types
from constants import date_index_label, signal_label
import pywt
import numpy as np
from sklearn.metrics import accuracy_score
from skimage.restoration import denoise_wavelet
from scipy.stats import uniform, randint
from sklearn.model_selection import ParameterSampler

without_pca_system = "without_pca"
with_pca_system = "with_pca"
with_pca_and_dwt_system = "with_pca_and_dwt"
system_types = [without_pca_system, with_pca_system, with_pca_and_dwt_system]


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


'''
def create_pipeline(system_type, **kwargs):
    if system_type == without_pca_system:
        normalizer = kwargs['normalizer']
        xgboost = kwargs['xgboost']
        steps = [('normalizer', normalizer), ('xgboost', xgboost)]
    
    if system_type == with_pca_system:
        normalizer = kwargs['normalizer']
        pca = kwargs['pca']
        xgboost = kwargs['xgboost']
        steps = [('normalizer', normalizer), ('pca', pca), ('xgboost', xgboost)]
    
    if system_type == with_pca_and_dwt_system:
        normalizer = kwargs['normalizer']
        pca = kwargs['pca']
        dwt = kwargs['dwt']
        xgboost = kwargs['xgboost']
        steps = [('normalizer', normalizer), ('pca', pca), ('dwt', dwt), ('xgboost', xgboost)]
    
    else:
        raise ValueError('Incorrect system type.')

    # Print the steps being applied in the pipeline
    print(f"Pipeline steps for {system_type}: {[step[0] for step in steps]}")
    return Pipeline(steps)
'''

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
    xgboost,
    seed,
):  
    param_search = {
        "max_depth": randint(3, 15),
        "learning_rate": uniform(0.01, 0.3),
        "min_child_weight": randint(1, 15),
        "subsample": uniform(0.5, 0.5)  # Corrected range to [0.5, 1.0)
    }
    
    n_iter_search = 50
    param_list = list(ParameterSampler(param_search, n_iter_search, random_state=seed))

    ### Preprocessor Pipeline Creation ###
    if system_type == with_pca_and_dwt_system:
        preprocessor = Pipeline([
            ('normalizer', MinMaxScaler()),
            ('pca', PCA(**pca_settings)),
            ('dwt', Wavelet(**dwt_params))
        ])
    elif system_type == with_pca_system:
        preprocessor = Pipeline([
            ('normalizer', MinMaxScaler()),
            ('pca', PCA(**pca_settings))
        ])
    else:  
        preprocessor = Pipeline([
            ('normalizer', MinMaxScaler())
        ])

    # Print the steps being applied in the preprocessor pipeline
    print(f"Preprocessor steps for {system_type}: {[step[0] for step in preprocessor.steps]}")

    preprocessor.fit(X_train)

    # Transform the datasets using the preprocessor
    X_train_transformed = preprocessor.transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    def evaluate_params(params, iteration):
        print(f"Iteration {iteration + 1}/{n_iter_search}")
        model = XGBClassifier(**xgboost, random_state=seed, **params)
        model.fit(X_train_transformed, Y_train,
                  eval_set=[(X_val_transformed, Y_val)], 
                  early_stopping_rounds=10,
                  verbose=False)
        y_val_pred = model.predict(X_val_transformed)
        return accuracy_score(Y_val, y_val_pred)

    # Perform the random search
    best_score = 0
    best_params = None

    for iteration, params in enumerate(param_list):
        score = evaluate_params(params, iteration)
        if score > best_score:
            best_score = score
            best_params = params

    print("Best parameters found: ", best_params)
    print("Best validation accuracy: ", best_score)

    ### Hyperparameter Tuning with Transformed Data ###
    best_model = XGBClassifier(**xgboost, random_state=seed, **best_params)
    best_model.fit(np.concatenate((X_train_transformed, X_val_transformed)), 
                   np.concatenate((Y_train, Y_val)))
    
    signal = best_model.predict(X_test_transformed)

    test_score = accuracy_score(Y_test, signal)
    print(f'System={system_type}, ticker={ticker}, test_score={test_score}')

    signal_df = pd.DataFrame({signal_label: signal}, index=X_test.index)
    return signal_df


def perform_modelling(
    tickers,
    tickers_train_path,
    tickers_val_path,
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

        val_df = pd.read_csv(
            f'{tickers_val_path}/{ticker}.csv', index_col=date_index_label
        )
        
        val_feat_df = pd.read_csv(
            f'{target_feature_path}/{ticker}.csv', index_col=date_index_label
        ).loc[val_df.index]

        test_df = pd.read_csv(
            f'{tickers_test_path}/{ticker}.csv', index_col=date_index_label
        )
        
        test_feat_df = pd.read_csv(
            f'{target_feature_path}/{ticker}.csv', index_col=date_index_label
        ).loc[test_df.index]

        common_features = train_feat_df.columns.intersection(val_feat_df.columns).intersection(test_feat_df.columns)
        train_feat_df = train_feat_df[common_features]
        val_feat_df = val_feat_df[common_features]
        test_feat_df = test_feat_df[common_features]

        for system_type in system_types:
            signal_df = _generate_test_signal(
                ticker,
                system_type,
                train_df,
                train_feat_df,
                val_df,
                val_feat_df,
                test_df,
                test_feat_df,
                pca_settings,
                dwt_params,
                xgboost,
                seed,
            )

            Path(f'{signal_path}/{system_type}').mkdir(exist_ok=True)
            signal_df.to_csv(f'{signal_path}/{system_type}/{ticker}.csv')


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers_train_path = config['data']['train_path']
    tickers_test_path = config['data']['test_path']
    tickers_val_path = config['data']['validation_path']
    target_feature_path = config['data']['target_var_path']
    tickers = config['tickers']
    signal_path = config['data']['signal_path']
    pca_settings = config['modelling']['pca']
    dwt_params = config['modelling']['dwt']
    xgboost = config['modelling']['xgboost']
    seed = config['seed']

    perform_modelling(
        tickers,
        tickers_train_path,
        tickers_val_path,
        tickers_test_path,
        target_feature_path,
        signal_path,
        pca_settings,
        dwt_params,
        xgboost,
        seed,
    )


if __name__ == '__main__':
    main()