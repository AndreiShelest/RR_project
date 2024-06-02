import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from strategies import TradingStrategy
from backtesting import Backtest
from constants import signal_label, strategy_metrics, accuracy_label, sharpe_label


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

    return
