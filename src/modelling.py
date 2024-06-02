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
    
class Optimizer_xgb(BaseEstimator, TransformerMixin):
    def __init__(self, n_individuals=128, n_generations=100, patience=10, cxpb=0.5, mutpb=0.2):
        self.n_individuals = n_individuals
        self.n_generations = n_generations
        self.patience = patience
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.best_params_ = None
    
    def fit(self, X, y=None):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0.01, 1.0)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            eta, gamma, max_depth, min_child_weight, subsample = individual
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'eta': eta,
                'gamma': gamma,
                'max_depth': int(max_depth * 10) + 1,
                'min_child_weight': min_child_weight * 10,
                'subsample': subsample,
                'n_estimators': 100
            }
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
            return (np.mean(scores),)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=0.01, up=1.0, eta=0.1, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)
        
        population = toolbox.population(n=self.n_individuals)
        hall_of_fame = tools.HallOfFame(1)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        class EarlyStopping:
            def __init__(self, patience=10):
                self.patience = patience
                self.counter = 0
                self.best_score = None
            
            def stop(self, score):
                if self.best_score is None or score > self.best_score:
                    self.best_score = score
                    self.counter = 0
                else:
                    self.counter += 1
                return self.counter >= self.patience
        
        early_stopping = EarlyStopping(patience=self.patience)
        
        for gen in range(self.n_generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb)
            fits = toolbox.map(toolbox.evaluate, offspring)
            
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            
            population = toolbox.select(offspring, len(population))
            
            best = tools.selBest(population, 1)[0]
            if early_stopping.stop(best.fitness.values[0]):
                print(f"Early stopping at generation {gen}")
                break
            
            if gen in [4, 6, 8]:
                toolbox.unregister("mutate")
                if gen == 4:
                    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.01, up=1.0, eta=0.1, indpb=0.3)
                elif gen == 6:
                    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.01, up=1.0, eta=0.1, indpb=0.35)
                elif gen == 8:
                    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.01, up=1.0, eta=0.1, indpb=0.4)
            
            record = stats.compile(population)
            print(f"Gen: {gen}, Record: {record}")
        
        hall_of_fame.update(population)
        best_individual = hall_of_fame[0]
        
        self.best_params_ = {
            'eta': best_individual[0],
            'gamma': best_individual[1],
            'max_depth': int(best_individual[2] * 10) + 1,
            'min_child_weight': best_individual[3] * 10,
            'subsample': best_individual[4],
            'n_estimators': 100
        }
        
        return self

    def transform(self, X, y=None):
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

class XGBoostClassifier(BaseEstimator):
    def __init__(self, booster='gbtree', objective='binary:logistic', eta=0.3, gamma=0, max_depth=6, 
                 min_child_weight=1, subsample=1, n_estimators=100):
        self.booster = booster
        self.objective = objective
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.n_estimators = n_estimators
        self.model = None
    
    def fit(self, X, y):
        self.model = xgb.XGBClassifier(
            booster=self.booster,
            objective=self.objective,
            eta=self.eta,
            gamma=self.gamma,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            n_estimators=self.n_estimators
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        y_pred = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_pred)
    
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
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    pca_settings,
    dwt_params,
    xgboost_settings,
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
    xgboost_settings,
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
