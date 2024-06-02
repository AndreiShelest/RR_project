import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from strategies import TradingStrategy
from backtesting import Backtest
from constants import (
    signal_label,
    strategy_metrics,
    accuracy_label,
    sharpe_label,
    date_index_label,
)
from deap import base, tools, creator
from system_types import create_pipeline, system_types
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from custom_estimators import Wavelet
import json
from strategy_execution import read_ticker_info_for_strategy


_learning_rate_label = 'learning_rate'
_max_depth_label = 'max_depth'
_min_child_weight = 'min_child_weight'
_subsample = 'subsample'

_xgboost_hyperparams = [
    _learning_rate_label,
    _max_depth_label,
    _min_child_weight,
    _subsample,
]


def _xgboost_fitness(
    ticker,
    model: XGBClassifier,
    X_train,
    Y_train: pd.DataFrame,
    X_val,
    Y_val: pd.DataFrame,
    ticker_val_df: pd.DataFrame,
    val_pd_index,
    strategy_settings,
):
    model.fit(X_train, Y_train)

    signal = model.predict(X_val)
    score = model.score(X_val, Y_val)

    signal_df = pd.DataFrame({signal_label: signal}, index=val_pd_index)
    signal_df.index = pd.to_datetime(signal_df.index)

    joined_df = ticker_val_df.join(signal_df, how='inner')

    backtest = Backtest(
        data=joined_df,
        strategy=TradingStrategy,
        cash=strategy_settings['initial_cash'],
        commission=strategy_settings['commissions'][ticker],
        trade_on_close=True,
    )
    results = backtest.run()
    results.rename(dict(strategy_metrics), inplace=True)

    accuracy = results[accuracy_label] / 100  # it is in percentage
    sharpe = results[sharpe_label]

    return (accuracy, sharpe)


def _generate_parameters(param_name, size, rng):
    if param_name == _learning_rate_label:
        return rng.random(size=size)
    if param_name == _max_depth_label:
        return rng.integers(low=0, high=20, size=size)
    if param_name == _min_child_weight:
        return rng.integers(low=0, high=100, size=size)
    if param_name == _subsample:
        return rng.random(size=size)

    raise f'Incorrect parameter: {param_name}'


def _generate_population(size, rng):
    gen_params = [
        _generate_parameters(param, size, rng) for param in _xgboost_hyperparams
    ]
    return [
        {
            _learning_rate_label: row[0],
            _max_depth_label: row[1],
            _min_child_weight: row[2],
            _subsample: row[3],
        }
        for row in zip(*gen_params)
    ]


def _unregister(toolbox, name):
    if hasattr(toolbox, name):
        toolbox.unregister(name)


def find_xgboost_hyperparameters(
    ticker,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    ticker_val_df,
    xgboost_settings,
    strategy_settings,
    transformer: Pipeline,
    seed,
    hp_config,
):
    Xm_train = transformer.fit_transform(X_train)
    Xm_val = transformer.transform(X_val)

    rng = np.random.default_rng(seed=seed)

    population = _generate_population(hp_config['pop_size'], rng)
    population_iterator = iter(population)

    toolbox = base.Toolbox()

    _unregister(toolbox, 'individual')
    _unregister(toolbox, 'population')
    _unregister(toolbox, 'evaluate')

    toolbox.register(
        'individual',
        tools.initIterate,
        creator.Individual,
        population_iterator.__next__,
    )
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        'evaluate',
        _xgboost_fitness,
        ticker=ticker,
        X_train=Xm_train,
        Y_train=Y_train,
        X_val=Xm_val,
        Y_val=Y_val,
        ticker_val_df=ticker_val_df,
        val_pd_index=X_val.index,
        strategy_settings=strategy_settings,
    )

    params_pop = toolbox.population(n=hp_config['pop_size'])

    for param_set in params_pop:
        combined_params = xgboost_settings | param_set
        xgboost = XGBClassifier(**combined_params)
        fit_values = toolbox.evaluate(model=xgboost)

        print(f'acc={fit_values[0]}, sharpe={fit_values[1]}')

    return


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    creator.create('xgboost_fitness', base.Fitness, weights=(1.0, 1.0))
    creator.create('Individual', dict, fitness=creator.xgboost_fitness)

    tickers = config['tickers']

    def get_features(ticker, path):
        return pd.read_csv(f'{path}/{ticker}.csv', index_col=date_index_label)

    def get_labels(ticker, feat_index):
        return pd.read_csv(
            f"{config['data']['target_var_path']}/{ticker}.csv",
            index_col=date_index_label,
        ).loc[feat_index]

    for ticker in tickers:
        X_train = get_features(ticker, config['data']['train_path'])
        X_train.dropna(inplace=True)
        Y_train = get_labels(ticker, X_train.index)

        X_val = get_features(ticker, config['data']['validation_path'])
        Y_val = get_labels(ticker, X_val.index)

        ticker_val_df = read_ticker_info_for_strategy(
            ticker, config['data']['validation_path']
        )

        for system_type in system_types:
            transformer = create_pipeline(
                system_type,
                normalizer=MinMaxScaler(),
                pca=PCA(**config['modelling']['pca']),
                dwt=Wavelet(**config['modelling']['dwt']),
            )

            print(f'Finding hyperparameters for system={system_type}, ticker={ticker}.')

            find_xgboost_hyperparameters(
                ticker,
                X_train,
                Y_train,
                X_val,
                Y_val,
                ticker_val_df,
                config['modelling']['xgboost'],
                config['strategy'],
                transformer,
                config['seed'],
                config['modelling']['hp_search'],
            )

            # Path(f'{signal_path}/{system_type}').mkdir(exist_ok=True)
            # signal_df.to_csv(f'{signal_path}/{system_type}/{ticker}.csv')


if __name__ == '__main__':
    main()
