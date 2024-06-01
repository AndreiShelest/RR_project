from backtesting import Strategy, Backtest
import json
import pandas as pd


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


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)
    tickers_test_path = config['data']['test_path']
    signal_path = config['data']['signal_path']
    tickers = config['tickers']

    for ticker in tickers:
        test_df = pd.read_csv(f'{tickers_test_path}/{ticker}.csv', index_col='Date')
        test_df.index = pd.to_datetime(test_df.index)

        signal_df = pd.read_csv(f'{signal_path}/{ticker}.csv', index_col='Date')
        signal_df.index = pd.to_datetime(signal_df.index)

        joined_df = test_df.join(signal_df, how='inner')

        print(
            f'[Strategy] Join kept all data: {len(test_df) == len(joined_df) and len(signal_df) == len(joined_df)}'
        )

        backtest = Backtest(
            data=joined_df, strategy=TradingStrategy, cash=10000, commission=0
        )
        results = backtest.run()
        print(
            f'Ticker={ticker}, Strategy={results["Return [%]"]}, B&H={results["Buy & Hold Return [%]"]}'
        )

    return


if __name__ == '__main__':
    main()
