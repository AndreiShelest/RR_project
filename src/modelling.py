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
import copy
import random
import numpy as np
from skimage.restoration import denoise_wavelet
import xgboost as xgb
from deap import base, creator, tools, algorithms
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


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
    
def sharpe_ratio(returns):
    # sharpe ratio
    mean_returns = np.mean(returns)
    std_returns = np.std(returns)
    return mean_returns / std_returns if std_returns != 0 else 0
    
class XGBoost_MOOGA(BaseEstimator, TransformerMixin):
    def __init__(self, n_individuals, n_generations, cxpb, mutpb, hypermutation1, hypermutation2):
        self.n_individuals = n_individuals
        self.n_generations = n_generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.hypermutation1 = hypermutation1
        self.hypermutation2 = hypermutation2
        self.best_params_ = None
        self.best_model_ = None

    def fit(self, X, y=None, X_val=None, y_val=None):
        # Initialize DEAP components
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # maximize both objectives
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.uniform, 0.01, 1.0)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_float, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def eval_individual(individual):
            params = {
                'learning_rate': individual[0],
                'max_depth': int(individual[1] * 10),
                'min_child_weight': individual[2] * 10,
                'subsample': individual[3]
            }
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                booster='gbtree',
                n_estimators=100,
                **params
            )
            model.fit(X, y)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            returns = y_pred - y_val  # Simplified return calculation, replace with actual trading returns
            sharpe = sharpe_ratio(returns)
            return accuracy, sharpe

        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=0.01, up=1.0, eta=0.1, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)

        population = toolbox.population(n=self.n_individuals)

        # Define the statistics and hall of fame
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        hof = tools.HallOfFame(1)

        # Variables for early stopping
        prev_best = None
        no_improvement_generations = 0

        # Evolutionary algorithm
        for gen in range(self.n_generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb)
            fits = toolbox.map(toolbox.evaluate, offspring)

            # Update fitness values
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            # Apply the selection process
            population = toolbox.select(offspring, k=len(population))
            record = stats.compile(population)
            print(f"Generation {gen}: {record}")

            # Update hall of fame
            hof.update(population)

            # Early stopping logic
            current_best = copy.deepcopy(hof.items[0])
            if prev_best is None or current_best.fitness.values != prev_best.fitness.values:
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1

            prev_best = copy.deepcopy(current_best)

            # Hypermutation logic
            if no_improvement_generations == 4:
                self.mutpb += self.hypermutation1
            elif no_improvement_generations == 6:
                self.mutpb += self.hypermutation2
            elif no_improvement_generations == 8:
                self.mutpb += self.hypermutation2
            elif no_improvement_generations >= 10:
                print("No improvement for 10 generations. Stopping early.")
                break

        # Extract the best parameters from the hall of fame
        self.best_params_ = {
            'learning_rate': hof[0][0],
            'max_depth': int(hof[0][1] * 10),
            'min_child_weight': hof[0][2] * 10,
            'subsample': hof[0][3]
        }

        # Train the final model with the best parameters on the entire dataset
        self.best_model_ = xgb.XGBClassifier(
            objective='binary:logistic',
            booster='gbtree',
            n_estimators=100,
            **self.best_params_
        )
        self.best_model_.fit(X, y)
        return self

    def transform(self, X):
        return self.best_model_.predict(X)
    
class Wavelet(BaseEstimator, TransformerMixin):

    def __init__(self, mode, level, wavelet) -> None:
        # self.ticker = ticker
        # self.df_index = df_index
        # self.interim_path = interim_path
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
        denoised_coeffs = [coeffs[0]]  # Keep approximation coefficients intact
        denoised_coeffs += [
            denoise_wavelet(c, method='BayesShrink', mode=self.mode, wavelet_levels=self.level, wavelet=self.wavelet, rescale_sigma=True)
            for c in coeffs[1:]]
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
    optimisation_param,

):
    model = create_pipeline(
        system_type,
        normalizer=MinMaxScaler(),
        pca=PCA(**pca_settings),
        dwt=Wavelet(**dwt_params),
        mooga = XGBoost_MOOGA(**optimisation_param),
    )
    model.fit(X_train, Y_train, X_val, Y_val)

    test_score = model.score(X_test, Y_test)
    print(f'System={system_type}, ticker={ticker}, test_score={test_score}')

    signal = model.predict(X_test)

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
    optimisation_param
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
                optimisation_param
            )

            Path(f'{signal_path}/{system_type}').mkdir(exist_ok=True)
            signal_df.to_csv(f'{signal_path}/{system_type}/{ticker}.csv')


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers_train_path = config['data']['train_path']
    tickers_val_path = config['data']['validation_path']
    tickers_test_path = config['data']['test_path']
    target_feature_path = config['data']['target_var_path']
    tickers = config['tickers']
    signal_path = config['data']['signal_path']
    pca_settings = config['modelling']['pca']
    dwt_params = config['modelling']['dwt']
    optimisation_param = config['modelling']['optimisation_param']


    perform_modelling(
        tickers,
        tickers_train_path,
        tickers_val_path,
        tickers_test_path,
        target_feature_path,
        signal_path,
        pca_settings,
        dwt_params,
        optimisation_param
    )


if __name__ == '__main__':
    main()
