from backtesting import Strategy, Backtest
import json
import pandas as pd

class TradingStrategy(Strategy):
    def init(self, indicator_func):
        return
    
    def next(self):
        return

def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)
    tickers_test_path = config['data']['test_path']
    tickers = config['tickers']

    for ticker in tickers:
        test_df =  pd.read_csv(f'{tickers_test_path}/{ticker}.csv', index_col='Date')
        test_df.index = pd.to_datetime(test_df.index)

        backtest = Backtest(data=test_df, strategy=TradingStrategy)
        results = backtest.run()
        # print(f'Ticker={ticker}, return={results["Buy & Hold Return [%]"]}')
    
    return

if __name__ == '__main__':
    main()