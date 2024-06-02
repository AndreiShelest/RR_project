import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from strategies import TradingStrategy
from backtesting import Backtest
from constants import signal_label, strategy_metrics, accuracy_label, sharpe_label
from deap import base, tools, creator

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

_n_ind = 128
_n_gen = 100


def _xgboost_fitness(
    ticker,
    model: XGBClassifier,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    strategy_settings,
):
    signal = model.predict(X_val)
    score = model.score(X_val, Y_val)

    signal_df = pd.DataFrame({signal_label: signal}, index=X_val.index)

    joined_df = X_val.join(signal_df, how='inner')

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

    return (accuracy, sharpe, score)


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


def find_xgboost_hyperparameters(
    ticker,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    xgboost_settings,
    strategy_settings,
    transformer: Pipeline,
    seed,
):
    Xm_train = transformer.fit_transform(X_train)
    Xm_val = transformer.transform(X_val)

    rng = np.random.default_rng(seed=seed)

    population = _generate_population(_n_ind, rng)
    population_iterator = iter(population)

    creator.create('xgboost_fitness', base.Fitness, weights=(1.0, 1.0))
    creator.create('Individual', dict, fitness=creator.xgboost_fitness)

    toolbox = base.Toolbox()

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
        X_val=Xm_val,
        Y_val=Y_val,
        strategy_settings=strategy_settings,
    )

    pop = toolbox.population(n=_n_ind)

    return
