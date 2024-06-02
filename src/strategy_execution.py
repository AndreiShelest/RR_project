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
    strategy_metrics,
    strategy_metrics_labels,
    return_label,
    bh_return_label,
)
from strategies import TradingStrategy, BuyAndHoldStrategy


def _validate_amount_of_trades(signal: pd.Series, trades_actual):
    shifted_signal = np.pad(
        signal, pad_width=(0, 1), constant_values=(0, signal.iat[-1])
    )[1:]
    trades_required = (np.abs(shifted_signal - signal) > 0).sum() + 1
    print(
        f'Trades required={trades_required}, trades actual={trades_actual}, valid={trades_required == trades_actual}'
    )


def _run_bh_strategy(test_df, initial_cash, commission):
    backtest = Backtest(
        data=test_df,
        strategy=BuyAndHoldStrategy,
        cash=initial_cash,
        commission=commission,
        trade_on_close=True,
    )
    results = backtest.run()
    results.rename(dict(strategy_metrics), inplace=True)

    return results

def read_ticker_info_for_strategy(ticker, path):
    ticker_df = pd.read_csv(
        f'{path}/{ticker}.csv', index_col=date_index_label
    )
    ticker_df.index = pd.to_datetime(ticker_df.index)
    ticker_df['Open'] = np.pad(
        ticker_df['Close'],
        pad_width=(1, 0),
        constant_values=(ticker_df['Close'].iat[0], np.NaN),
    )[: len(ticker_df)]
    ticker_df['High'] = ticker_df['Close']
    ticker_df['Low'] = ticker_df['Close']

    return ticker_df

def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)
    tickers_test_path = config['data']['test_path']
    signal_path = config['data']['signal_path']
    basic_stats_path = config['data']['basic_stats_path']
    ts_stats_path = config['data']['time_series_stats_path']
    tickers = config['tickers']
    commissions = config['strategy']['commissions']
    initial_cash = config['strategy']['initial_cash']

    basic_stats_data = {}
    time_series_data = {}

    def assign_basic_stats(system_type, ticker, results):
        if basic_stats_data.get(system_type) is None:
            basic_stats_data[system_type] = pd.DataFrame(index=strategy_metrics_labels)
            basic_stats_data[system_type].index.name = 'Metric'
        basic_stats_data[system_type][ticker] = results

    def assign_ts_stats(system_type, ticker, results, initial_cash):
        if time_series_data.get(equity_label) is None:
            time_series_data[equity_label] = {}
        if time_series_data[equity_label].get(ticker) is None:
            time_series_data[equity_label][ticker] = pd.DataFrame()

        # shift it since backtesting.py saves the equity of the next day in the previous day somehow
        original_ec = results[equity_curve_label][equity_label]
        shifted_ec = np.pad(
            original_ec, pad_width=(1, 0), constant_values=(initial_cash, np.NaN)
        )[: len(original_ec)]
        shifted_ec = pd.Series(shifted_ec, index=original_ec.index)

        time_series_data[equity_label][ticker][system_type] = shifted_ec

    for ticker in tickers:
        test_df = read_ticker_info_for_strategy(ticker, tickers_test_path)

        commission = commissions[ticker]

        bh_results = _run_bh_strategy(test_df, initial_cash, commission)
        assign_basic_stats(buy_hold_system, ticker, bh_results)
        assign_ts_stats(buy_hold_system, ticker, bh_results, initial_cash)

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
                cash=initial_cash,
                commission=commission,
                trade_on_close=True,
            )
            results = backtest.run()
            results.rename(dict(strategy_metrics), inplace=True)

            assign_basic_stats(system_type, ticker, results)
            assign_ts_stats(system_type, ticker, results, initial_cash)

            print(
                f'[{system_type}] Ticker={ticker}, Strategy Return={results[return_label]}, B&H={results[bh_return_label]}'
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
