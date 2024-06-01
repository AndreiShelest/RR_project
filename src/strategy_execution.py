from backtesting import Strategy, Backtest
import json
import pandas as pd
from system_types import system_types
from pathlib import Path


class TradingStrategy(Strategy):
    idx = 0
    prev_signal = 0

    def init(self):
        super().init()

    def next(self):
        curr_signal = self.data['Signal'][self.idx]
        if curr_signal == 1 and self.prev_signal == 0:
            self.buy()
        elif curr_signal == 0 and self.prev_signal == 1:
            self.sell()

        self.idx += 1
        self.prev_signal = curr_signal


_return_label = "Return"
_return_pa_label = "Return (p.a.)"
_vol_pa_label = "Vol (p.a.)"
_bh_return_label = "B&H Return"
_sharpe_label = "Sharpe Ratio"
_mdd_label = "Max Drawdown"
_accuracy_label = "Accuracy"
_transactions_label = "Transactions"

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


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)
    tickers_test_path = config['data']['test_path']
    signal_path = config['data']['signal_path']
    basic_stats_path = config['data']['basic_stats_path']
    tickers = config['tickers']

    basic_stats_data = {}

    for ticker in tickers:
        test_df = pd.read_csv(f'{tickers_test_path}/{ticker}.csv', index_col='Date')
        test_df.index = pd.to_datetime(test_df.index)

        for system_type in system_types:
            signal_df = pd.read_csv(
                f'{signal_path}/{system_type}/{ticker}.csv', index_col='Date'
            )
            signal_df.index = pd.to_datetime(signal_df.index)

            joined_df = test_df.join(signal_df, how='inner')

            print(
                f'[{system_type}] Join kept all data: {len(test_df) == len(joined_df) and len(signal_df) == len(joined_df)}'
            )

            backtest = Backtest(
                data=joined_df, strategy=TradingStrategy, cash=100000, commission=0
            )
            results = backtest.run()

            results.rename(dict(_strategy_metrics), inplace=True)

            if basic_stats_data.get(system_type) is None:
                basic_stats_data[system_type] = pd.DataFrame(
                    index=_strategy_metrics_labels
                )
                basic_stats_data[system_type].index.name = 'Metric'
            basic_stats_data[system_type][ticker] = results

            print(
                f'[{system_type}] Ticker={ticker}, Strategy Return={results[_return_label]}, B&H={results[_bh_return_label]}'
            )

    for system_type in basic_stats_data:
        basic_stats_data[system_type].to_csv(f'{basic_stats_path}/{system_type}.csv')

    return


if __name__ == '__main__':
    main()
