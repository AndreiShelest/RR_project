from talib import abstract as t_abs
import pandas as pd
import json

SMA = t_abs.Function('SMA')
EMA = t_abs.Function('EMA')
BBANDS = t_abs.Function('BBANDS')
STOCH = t_abs.Function('STOCH')
MACD = t_abs.Function('MACD')

technical_indicators = [
    ('RSI', t_abs.Function('RSI')),
    ('PPO', t_abs.Function('PPO')),
    ('ADX', t_abs.Function('ADX')),
    ('MOM', t_abs.Function('MOM')),
    ('CCI', t_abs.Function('CCI')),
    ('ROC', t_abs.Function('ROC')),
    ('WILLR', t_abs.Function('WILLR')),
    ('SMA20', lambda df: SMA(df, timeperiod=20)),
    ('SMA50', lambda df: SMA(df, timeperiod=50)),
    ('SMA100', lambda df: SMA(df, timeperiod=100)),
    ('EMA20', lambda df: EMA(df, timeperiod=20)),
    ('EMA50', lambda df: EMA(df, timeperiod=50)),
    ('EMA100', lambda df: EMA(df, timeperiod=100)),
    ('PSAR', t_abs.Function('SAR')),
    ('OBV', t_abs.Function('OBV')),
    ('ADOSC', t_abs.Function('ADOSC')),
    ('MFI', t_abs.Function('MFI')),
    ('ATR', t_abs.Function('ATR')),
]

def _perform_technical_analysis(ticker_df: pd.DataFrame):
    for indicator, func in technical_indicators:
        ticker_df[indicator] = func(ticker_df)

    bbands_results = BBANDS(ticker_df)
    ticker_df['BBANDS_upper'], ticker_df['BBANDS_middle'], ticker_df['BBANDS_lower'] = \
        bbands_results.upperband, bbands_results.middleband, bbands_results.lowerband

    stoch_results = STOCH(ticker_df)
    ticker_df['STOCH_K'], ticker_df['STOCK_D'] = \
        stoch_results.slowk, stoch_results.slowd

    macd_results = MACD(ticker_df)
    ticker_df['MACD'], ticker_df['MACD_S'], ticker_df['MACD_H'] = \
        macd_results.macd, macd_results.macdsignal, macd_results.macdhist

def _prepare_ticker_df(ticker_df: pd.DataFrame):
    ticker_df.drop(columns=['Close', 'Dividends', 'Stock Splits'], inplace=True)

    ticker_df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Adj Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    ticker_df.index.name = 'date'

def perform_ta(tickers: list[str], input_dir_path: str, output_dir_path: str):
    ticker_dfs = [(ticker, pd.read_csv(f'{input_dir_path}/{ticker}.csv',
                              index_col='Date')) for ticker in tickers]
    
    for ticker, ticker_df in ticker_dfs:
        _prepare_ticker_df(ticker_df)
        _perform_technical_analysis(ticker_df)

        ticker_df.to_csv(f'{output_dir_path}/{ticker}.csv')

        print(f'TA for {ticker} has been performed.')

    return ticker_dfs

def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers_dir_path = config['data']['tickers_path']
    tickers_ta_dir_path = config['data']['tickers_ta_path']
    tickers = config['tickers']

    perform_ta(tickers, tickers_dir_path, tickers_ta_dir_path)

if __name__ == '__main__':
    main()