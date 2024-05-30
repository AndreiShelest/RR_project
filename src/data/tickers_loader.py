import yfinance as yf
from datetime import datetime, timedelta
import json


def _load_ticker_history(ticker: yf.Ticker, start_date, end_date):
    print(f'Downloading of {ticker.ticker} is starting.')
    return ticker.history(
        start=start_date, end=end_date + timedelta(days=1), auto_adjust=False
    )


def _download_historical_data(
    tickers: list[str], start_date: datetime, end_date: datetime
):
    tickers_data = yf.Tickers(tickers)

    histories = [
        (
            ticker,
            _load_ticker_history(
                (yf_ticker := tickers_data.tickers[ticker]), start_date, end_date
            ),
            yf_ticker.info['shortName'],
        )
        for ticker in tickers
    ]

    return histories


def load_and_store(
    tickers: list[str], start_date: datetime, end_date: datetime, folder_path: str
):
    histories = _download_historical_data(tickers, start_date, end_date)

    for ticker, history, _ in histories:
        history.to_csv(f'{folder_path}/{ticker}.csv', date_format='%Y-%m-%d')

        print(f'Ticker info {ticker} is loaded and stored.')

    return [(ticker, name) for ticker, _, name in histories]


def main():
    with open('./project_config.json', 'r') as config_file:
        config = json.load(config_file)

    tickers = config['tickers']
    start_date = datetime.fromisoformat(config['start_date'])
    end_date = datetime.fromisoformat(config['end_date'])
    tickers_dir_path = config['data']['tickers_path']

    load_and_store(tickers, start_date, end_date, tickers_dir_path)


if __name__ == '__main__':
    main()
