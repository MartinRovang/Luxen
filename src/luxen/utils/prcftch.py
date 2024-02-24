import pandas
from pandas_datareader import data as pdr
import yfinance as yfin
from loguru import logger
from typing import List
from pypfopt import expected_returns, risk_models, objective_functions, EfficientFrontier
import matplotlib.pyplot as plt
import mplcyberpunk
import pkg_resources
import pandas as pd
plt.style.use("cyberpunk")

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
        symbols = [x + ".OL" for x in symbols]
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

    def fetch_all_adj_close(self) -> pandas.DataFrame:
        """Fetches the data from Yahoo Finance
        Returns:
            pandas.DataFrame: Dataframe with the data
            None: If there was an error
        """
        df = self.fetch()
        if df is None:
            raise Exception("Error fetching data")
        # clean remove all nan columns
        df.dropna(axis=1, inplace=True)
        return df["Adj Close"]

    def rebalance_portofolio(self, portofolio_target: pandas.DataFrame, stock_num:  pandas.DataFrame, weights_started: pandas.DataFrame) -> pandas.DataFrame:
        """Rebalances the portofolio based on the weights
        Finds how much to sell or buy of a particular stock based on the weights to restore it to the desired weights
        Args:
            portofolio_target (pandas.DataFrame): The target portofolio
            stock_num (pandas.DataFrame): The number of stocks
            weights_started (pandas.DataFrame): The weights of the portofolio
        Returns:
            pandas.DataFrame: The rebalanced portofolio
        """
        # to pct
        portofolio_target_weighted = portofolio_target * stock_num.iloc[0]

        # remove nan
        # portofolio_target_pct_weighted.dropna(inplace=True)
        # portofolio_started_pct_weighted.dropna(inplace=True)

        # get new weighting of the target portofolio
        current_weighting = portofolio_target_weighted.iloc[-1] / portofolio_target_weighted.iloc[-1].sum()

        # find rebalancing weights
        weights = weights_started - current_weighting

        # find the amount of stocks to buy or sell
        rebalanced_stocks = round(stock_num * weights)

        return rebalanced_stocks


if __name__ == "__main__":
    pf = PriceFetcher(start_date="2019-01-01", end_date="2024-02-23")
    pf.use_tickers_from_file(r"C:\Users\Gimpe\Documents\github\Luxen\src\luxen\data\Euronext_Equities_2024-01-03.csv")
    all_stock = pf.fetch_all_adj_close()

    # # 'BELCO.OL', 0.09451), ('KID.OL', 0.09882), ('AGAS.OL', 0.07804), ('YAR.OL', 0.10904), ('KOG.OL', 0.10771), ('VISTN.OL', 0.0994), ('ABL.OL', 0.09604), ('WWI.OL', 0.10364), ('ENTRA.OL', 0.10548), ('SBO.OL', 0.10731)])
    # given_portofolio = all_stock[['BELCO.OL', 'KID.OL', 'AGAS.OL', 'YAR.OL', 'KOG.OL', 'VISTN.OL', 'ABL.OL', 'WWI.OL', 'ENTRA.OL', 'SBO.OL']]

    # stock_num = {
    #     "BELCO.OL": 54,
    #     "KID.OL": 43,
    #     "AGAS.OL": 114,
    #     "YAR.OL": 60,
    #     "KOG.OL": 90,
    #     "VISTN.OL": 100,
    #     "ABL.OL": 10,
    #     "WWI.OL": 200,
    #     "ENTRA.OL": 70,
    #     "SBO.OL": 95
    # }
    # weights_started = {
    #     "BELCO.OL": 0.09451,
    #     "KID.OL": 0.09882,
    #     "AGAS.OL": 0.07804,
    #     "YAR.OL": 0.10904,
    #     "KOG.OL": 0.10771,
    #     "VISTN.OL": 0.0994,
    #     "ABL.OL": 0.09604,
    #     "WWI.OL": 0.10364,
    #     "ENTRA.OL": 0.10548,
    #     "SBO.OL": 0.10731
    # }
    # # into a dataframe set as rows
    # stock_num = pandas.DataFrame(stock_num, index=[0])
    # weights_started = pandas.DataFrame(weights_started, index=[0])

    # # rebalance the portofolio
    # rebalanced_stocks = pf.rebalance_portofolio(portofolio_target=given_portofolio, stock_num=stock_num, weights_started=weights_started)
    # # print(rebalanced_stocks)
    # # exit()


    annual_return_best, annual_volatility_best, sharpe_best, best_portofolio_weights = 0, 0, 0, None
    for i in range(0, 100000):
        current_portofolio = all_stock.sample(n=10, replace=False, axis=1)
        mu = expected_returns.mean_historical_return(current_portofolio)
        S = risk_models.risk_matrix(prices=current_portofolio, method="ledoit_wolf")
        ef = EfficientFrontier(mu, S, weight_bounds=[0.05, 1])
        ef.add_objective(objective_functions.L2_reg, gamma=1)
        ef.min_volatility()
        annual_return, annual_volatility, sharpe = ef.portfolio_performance(verbose=False)
        weights = ef.clean_weights()

        if float(sharpe_best) < float(sharpe):
            annual_return_best = annual_return
            annual_volatility_best = annual_volatility
            sharpe_best = sharpe
            best_portofolio_weights = weights
            best_portofolio = current_portofolio

            # ('STOCK.OL', WEIGHTs)
            weights_dic = {k: v for k, v in weights.items()}
            # into df
            weights_df = pd.DataFrame(weights_dic, index=[0])

            num_of_stocks_to_buy = (50000 / best_portofolio.iloc[-1]) * weights_df
            num_to_buy = round(num_of_stocks_to_buy)

            # plot the weighted portofolio time series
            diff = best_portofolio.pct_change()
            diff = (diff * best_portofolio_weights).sum(axis=1)
            # cumsum
            diff = (1 + diff).cumprod()
            diff.plot()

            # add boilinger bands
            # rolling = diff.rolling(window=20)
            # rolling_mean = rolling.mean()
            # rolling_std = rolling.std()
            # plt.plot(rolling_mean, label="Rolling mean", color="red", alpha=0.5)
            # plt.plot(rolling_mean + 2 * rolling_std, label="Bollinger band", color="yellow", alpha=0.5)
            # plt.plot(rolling_mean - 2 * rolling_std, label="Bollinger band", color="yellow", alpha=0.5)

            plt.title(f"sharpe: {sharpe_best:.2f} | mean: {annual_return_best:.2%} | std: {annual_volatility_best:.2%}")
            mplcyberpunk.add_glow_effects()
            plt.savefig(f"./examples/weighted_portofolio_{sharpe_best}.png")
            plt.close()

            # save meta data as text file
            with open(f"./examples/weighted_portofolio_{sharpe_best}.txt", "w") as f:
                f.write(f'Best portofolio:\n{best_portofolio_weights}\n')
                f.write(f'Annual return:\n {annual_return_best:.2%}, Annual volatility: {annual_volatility_best:.2%}, Sharpe Ratio: {sharpe_best:.2f}\n')
                f.write(f'\nNumber of stocks to buy:\n{num_to_buy}')