from backtesting import Backtest
import json
import pandas as pd
from system_types import system_types
from pathlib import Path
import numpy as np
from constants import (
    date_index_label,
    signal_label,
    buy_hold_system,
    equity_label,
    equity_curve_label,
)
from strategies import TradingStrategy, BuyAndHoldStrategy


_return_label = "Return"
_return_pa_label = "Return (p.a.)"
_vol_pa_label = "Vol (p.a.)"
_bh_return_label = "B&H Return"
_sharpe_label = "Sharpe Ratio"
_mdd_label = "Max Drawdown"
_accuracy_label = "Accuracy"
_transactions_label = "Positions Taken"

_strategy_metrics = [
    ("Return [%]", _return_label),
    ("Return (Ann.) [%]", _return_pa_label),
    ("Volatility (Ann.) [%]", _vol_pa_label),
    ("Buy & Hold Return [%]", _bh_return_label),
    ("Sharpe Ratio", _sharpe_label),
    ("Max. Drawdown [%]", _mdd_label),
    ("Win Rate [%]", _accuracy_label),
    ("# Trades", _transactions_label),
]
_strategy_metrics_labels = [label_pair[1] for label_pair in _strategy_metrics]


def _validate_amount_of_trades(signal: pd.Series, trades_actual):
    shifted_signal = np.pad(
        signal, pad_width=(0, 1), constant_values=(0, signal.iat[-1])
    )[1:]
    trades_required = (np.abs(shifted_signal - signal) > 0).sum() // 2 + (
        1 if signal.iat[0] == 1 else 0
    )
    print(
        f'Trades required={trades_required}, trades actual={trades_actual}, valid={trades_required == trades_actual}'
    )


def _run_bh_strategy(test_df):
    backtest = Backtest(
        data=test_df,
        strategy=BuyAndHoldStrategy,
        cash=100000,
        commission=0,
        trade_on_close=True,
    )
    results = backtest.run()
    results.rename(dict(_strategy_metrics), inplace=True)

    return results


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)
    tickers_test_path = config['data']['test_path']
    signal_path = config['data']['signal_path']
    basic_stats_path = config['data']['basic_stats_path']
    ts_stats_path = config['data']['time_series_stats_path']
    tickers = config['tickers']

    basic_stats_data = {}
    time_series_data = {}

    def assign_basic_stats(system_type, ticker, results):
        if basic_stats_data.get(system_type) is None:
            basic_stats_data[system_type] = pd.DataFrame(index=_strategy_metrics_labels)
            basic_stats_data[system_type].index.name = 'Metric'
        basic_stats_data[system_type][ticker] = results

    def assign_ts_stats(system_type, ticker, results):
        if time_series_data.get(equity_label) is None:
            time_series_data[equity_label] = {}
        if time_series_data[equity_label].get(ticker) is None:
            time_series_data[equity_label][ticker] = pd.DataFrame()
        time_series_data[equity_label][ticker][system_type] = results[
            equity_curve_label
        ][equity_label]

    for ticker in tickers:
        test_df = pd.read_csv(
            f'{tickers_test_path}/{ticker}.csv', index_col=date_index_label
        )
        test_df.index = pd.to_datetime(test_df.index)
        test_df['Open'] = test_df['Close']
        test_df['High'] = test_df['Close']
        test_df['Low'] = test_df['Close']

        bh_results = _run_bh_strategy(test_df)
        assign_basic_stats(buy_hold_system, ticker, bh_results)
        assign_ts_stats(buy_hold_system, ticker, bh_results)

        for system_type in system_types:
            signal_df = pd.read_csv(
                f'{signal_path}/{system_type}/{ticker}.csv', index_col=date_index_label
            )
            signal_df.index = pd.to_datetime(signal_df.index)

            joined_df = test_df.join(signal_df, how='inner')
            print(
                f'[{system_type}] Join kept all data: {len(test_df) == len(joined_df) and len(signal_df) == len(joined_df)}'
            )

            backtest = Backtest(
                data=joined_df,
                strategy=TradingStrategy,
                cash=100000,
                commission=0,
                trade_on_close=True,
            )
            results = backtest.run()
            results.rename(dict(_strategy_metrics), inplace=True)

            assign_basic_stats(system_type, ticker, results)
            assign_ts_stats(system_type, ticker, results)

            print(
                f'[{system_type}] Ticker={ticker}, Strategy Return={results[_return_label]}, B&H={results[_bh_return_label]}'
            )

            _validate_amount_of_trades(signal_df[signal_label], len(results['_trades']))

    for system_type in basic_stats_data:
        basic_stats_data[system_type].to_csv(f'{basic_stats_path}/{system_type}.csv')

    for ts_label in time_series_data:
        Path(f'{ts_stats_path}/{ts_label}').mkdir(exist_ok=True)

        for ticker in time_series_data[ts_label]:
            time_series_data[ts_label][ticker].to_csv(
                f'{ts_stats_path}/{ts_label}/{ticker}.csv'
            )

    return


if __name__ == '__main__':
    main()
