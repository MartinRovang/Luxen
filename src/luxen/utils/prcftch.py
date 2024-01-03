import pandas
from pandas_datareader import data as pdr
import yfinance as yfin
from loguru import logger
from typing import List

yfin.pdr_override()


class PriceFetcher:
    def __init__(self, start_date: str, end_date: str, tickers: List = [str]):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def use_tickers_from_file(self, file_path: str) -> None:
        """Reads the tickers from a file and sets them as the tickers to fetch
        Args:
            file_path (str): Path to the file
        """
        # read csv
        df = pandas.read_csv(file_path, delimiter=";")
        symbols = df["Symbol"].tolist()
        # remove all nans
        symbols = [x for x in symbols if str(x) != "nan"]
        self.tickers = symbols

    def fetch(self) -> pandas.DataFrame | None:
        """Fetches the data from Yahoo Finance
        Returns:
            pandas.DataFrame: Dataframe with the data
            None: If there was an error
        """
        try:
            df = pdr.get_data_yahoo(self.tickers, start=self.start_date, end=self.end_date)
            return df
        except Exception as e:
            logger.error(f"Error fetching {self.tickers}: {e}")
            return None

    def fetch_all_adj_close(self) -> pandas.DataFrame | None:
        """Fetches the data from Yahoo Finance
        Returns:
            pandas.DataFrame: Dataframe with the data
            None: If there was an error
        """
        df = self.fetch()
        if df is None:
            return None
        # clean remove all nan columns
        df.dropna(axis=1, inplace=True)
        return df["Adj Close"]


if __name__ == "__main__":
    pf = PriceFetcher(start_date="2023-01-01", end_date="2023-01-31")
    pf.use_tickers_from_file(r"C:\Users\Gimpe\Documents\github\Luxen\src\luxen\data\Euronext_Equities_2024-01-03.csv")
    print(pf.fetch_all_adj_close())