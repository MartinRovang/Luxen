from collections import defaultdict
import datetime
from numpy.random import f
import pandas
from pandas_datareader import data as pdr
import yfinance as yfin
from loguru import logger
from typing import List, OrderedDict
from pypfopt import expected_returns, risk_models, objective_functions, EfficientFrontier, black_litterman, BlackLittermanModel
import matplotlib.pyplot as plt
import mplcyberpunk
import pkg_resources
import numpy as np
import pandas as pd
import warnings
plt.style.use("cyberpunk")

yfin.pdr_override()



def fxn():
    warnings.warn("deprecated", DeprecationWarning)

class PriceFetcher:
    def __init__(self, start_date: str, end_date: str, tickers: List = [str]):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def use_tickers_from_file(
        self,
        file_path1: str,
        file_path2: str
    ) -> None:
        """Reads the tickers from a file and sets them as the tickers to fetch
        Args:
            file_path (str): Path to the file
        """
        # read csv

        # SEK => ST , NOK => OL , EUR => HE, DKK => CO
        # https://live.euronext.com/nb/products/equities
        df = pandas.read_csv(file_path1, delimiter=";")
        symbols = df["Symbol"].tolist()
        # remove all nans
        symbols = [x for x in symbols if str(x) != "nan"]
        symbols = [x + ".OL" for x in symbols]
        self.tickers = symbols
        

        # read file2.txt
        # TRATON	8TRA	SEK	DE000TRAT0N7	Industrials	5020	
        # AAK	AAK	SEK	SE0011337708	Consumer Goods	4510
        # https://www.nasdaqomxnordic.com/shares/listed-companies/nordic-large-cap
        columns = ["Symbol", "Currency", "ISIN", "Sector", "Industry", "Nan"]
        df = pandas.read_csv(file_path2, delimiter="\t", names=columns)

        # if Currency is SEK, add .ST
        symbols_ = df[df["Currency"] == "SEK"]["Symbol"].tolist()
        symbols = [x.replace(" ", "-") + ".ST" for x in symbols_]

        self.tickers += symbols

        # if Currency is NOK, add .OL
        symbols_ += df[df["Currency"] == "NOK"]["Symbol"].tolist()
        symbols = [x.replace(" ", "-") + ".OL" for x in symbols_]

        self.tickers += symbols

        # if Currency is EUR, add .HE
        symbols_ += df[df["Currency"] == "EUR"]["Symbol"].tolist()
        symbols = [x.replace(" ", "-") + ".HE" for x in symbols_]

        self.tickers += symbols

        # if Currency is DKK, add .CO
        symbols_ += df[df["Currency"] == "DKK"]["Symbol"].tolist()
        symbols = [x.replace(" ", "-") + ".CO" for x in symbols_]

        self.tickers += symbols

        # remove all that include "-"
        self.tickers = [x for x in self.tickers if "-" not in x]


        #self.tickers = self.tickers[50:52]

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
        save_spot = "/home/martin/Documents/github/Luxen/src/luxen/data/adj_close.csv"
        if pkg_resources.resource_exists("luxen", "data/adj_close.csv"):
            df = pandas.read_csv(save_spot)
            df.set_index("Date", inplace=True)
            # date to datetime
            df.index = pd.to_datetime(df.index)
            return df

        df = self.fetch()
        if df is None:
            raise Exception("Error fetching data")
        # return adj close and date
        payload = df["Adj Close"]

        # drop all columns with nans (stocks that are not available), but do not drop all columns
        payload.dropna(axis=1, how="all", inplace=True)

        # save to csv
        payload.to_csv("C:/Users/Gimpe/Documents/github/Luxen/src/luxen/data/adj_close.csv")

        return payload

    # def simulate_portofolio_rebalance_ahead(
    #     self, all_stock: pandas.DataFrame,
    #     weights: OrderedDict,
    #     input_credits: float = 100000,
    #     rebalance_every: int = 30*3,
    #     buy_date: datetime.datetime = pd.to_datetime('2020-01-23'),
    #     show_plot: bool = True
    #     ) -> float:
    #     """Simulates the portofolio ahead

    #     Args:
    #         all_stock (pandas.DataFrame): The stocks
    #         weights (OrderedDict): The weights
    #         input_credits (float, optional): The input credits. Defaults to 100000.
    #         rebalance_every (int, optional): How often to rebalance the portofolio. Defaults to 30*3.
    #         buy_date (str, optional): The date to buy the stocks. Defaults to "2022-02-23".
    #         show:plot (bool, optional): Whether to show the plot. Defaults to True.

    #     Returns:
    #         float: The total gain of the portofolio
    #     """
    #     # plot the weighted portofolio time series
    #     portofolio = all_stock[list(weights.keys())]
    #     portofolio_chunked_rebalanced = []
    #     for i in range(0, len(portofolio), rebalance_every):
    #         portofolio_chunk = portofolio.iloc[i:i+rebalance_every]
    #         diff = portofolio_chunk.pct_change.(fill_method=None).dropna(how="all")
    #         diff = (diff * weights).sum(axis=1)

    #         # cumsum
    #         diff = (1 + diff).cumprod() * input_credits
    #         portofolio_chunked_rebalanced.append(diff)


    #     # find how many stocks bought
    #     stock_num = (input_credits / portofolio.iloc[0]).round()


    #     # find the rebalanced stocks
    #     rebalanced_stocks = self.rebalance_portofolio(portofolio_target=portofolio, stock_num=stock_num, weights_started=weights)


    #     portofolio = portofolio.loc[buy_date:]
    #     diff = portofolio.pct_change(fill_method=None).dropna(how="all")
    #     diff = (diff * weights).sum(axis=1)

    #     # cumsum
    #     diff = (1 + diff).cumprod() * input_credits
    #     if show_plot:
    #         diff.plot()
    #         plt.title("Portofolio simulation")
    #         mplcyberpunk.add_glow_effects()
    #         plt.show()

    #     total_gain = round(diff.iloc[-1] - input_credits)
    #     return total_gain

    def simulate_portofolio_ahead(
        self,
        all_stock: pandas.DataFrame,
        weights: OrderedDict,
        input_credits: float = 100000,
        buy_date: datetime.datetime = pd.to_datetime('2020-01-23'),
        show_plot: bool = True
    ) -> float:
        """Simulates the portofolio ahead
        Args:
            all_stock (pandas.DataFrame): The stocks
            weights (OrderedDict): The weights
            input_credits (float, optional): The input credits. Defaults to 100000.
            buy_date (str, optional): The date to buy the stocks. Defaults to "2022-02-23".
            show:plot (bool, optional): Whether to show the plot. Defaults to True.
        Returns:
            float: The total gain of the portofolio
        """
        # plot the weighted portofolio time series
        portofolio = all_stock[list(weights.keys())]

        portofolio = portofolio.loc[buy_date:]
        diff = portofolio.pct_change()
        diff = (diff * weights).sum(axis=1)

        # cumsum
        diff = (1 + diff).cumprod() * input_credits
        if show_plot:
            diff.plot()
            plt.title("Portofolio simulation")
            mplcyberpunk.add_glow_effects()
            plt.show()

        total_gain = round(diff.iloc[-1] - input_credits)
        return total_gain

    def rebalance_portofolio(
        self,
        portofolio_target: pandas.DataFrame,
        stock_num: pandas.DataFrame,
        weights_started: OrderedDict
    ) -> pandas.DataFrame:
        """Rebalances the portofolio based on the weights
        Finds how much to sell or buy of a particular stock based on the weights to restore it to the desired weights
        Args:
            portofolio_target (pandas.DataFrame): The target portofolio
            stock_num (pandas.DataFrame): The number of stocks
            OrderedDict (OrderedDict): The weights of the portofolio
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

    def target_portofolio_plot(self, all_stock: pandas.DataFrame, weights: pandas.DataFrame, title: str) -> None:
        """Plots the target portofolio
        Args:
            all_stock (pandas.DataFrame): The stocks
            weights (pandas.DataFrame): The weights
            title (str): The title of the plot
        """
        # plot the weighted portofolio time series
        portofolio = all_stock[list(weights.keys())]

        diff = portofolio.pct_change(fill_method=None).dropna(how="all")
        diff = (diff * weights).sum(axis=1)
        # cumsum
        diff = (1 + diff).cumprod()
        diff.plot()

        plt.title(title)
        mplcyberpunk.add_glow_effects()
        plt.show()


if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

        cash = 50000
        num_of_iterations = 50000
        N = 10
        buy_date = pd.to_datetime('2020-01-02')
        sample_for_real_portofolio: bool = True

        start_date = "2020-01-01"
        end_date = "2024-01-25"
        years = pd.date_range(start=buy_date, end=end_date, freq="Y")
        total_years = len(years)

        pf = PriceFetcher(start_date=start_date, end_date=end_date)
        file1 = "/home/martin/Documents/github/Luxen/src/luxen/data/Euronext_Equities_2024-02-25.csv"
        file2 = "/home/martin/Documents/github/Luxen/src/luxen/data/scandi.txt"
        pf.use_tickers_from_file(file1, file2)
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

        #OrderedDict([('GYL.OL', 0.12277), ('REACH.OL', 0.09312), ('NHY.OL', 0.10896), ('HAVI.OL', 0.05905), ('SBO.OL', 0.12388), ('EIOF.OL', 0.08913), ('AZT.OL', 0.09279), ('NAPA.OL', 0.07381), ('SRBNK.OL', 0.11882), ('WWI.OL', 0.11767)])
        #weights_started = eval("OrderedDict([('GYL.OL', 0.12277), ('REACH.OL', 0.09312), ('NHY.OL', 0.10896), ('HAVI.OL', 0.05905), ('SBO.OL', 0.12388), ('EIOF.OL', 0.08913), ('AZT.OL', 0.09279), ('NAPA.OL', 0.07381), ('SRBNK.OL', 0.11882), ('WWI.OL', 0.11767)])")

        #pf.target_portofolio_plot(all_stock=all_stock, weights=weights_started, title="Target portofolio")
        #pf.simulate_portofolio_ahead(all_stock=all_stock, weights=weights_started, input_credits=100000, buy_date=pd.to_datetime('2020-01-23'))

        # find the best portofolio
        annual_return_best, annual_volatility_best, sharpe_best, best_portofolio_weights = 0, 0, 1, None
        earned_money_all = []
        num_of_fail_and_success = defaultdict(int)
        for i in range(0, num_of_iterations):
            current_portofolio = all_stock.sample(n=N, replace=False, axis=1)
            current_portofolio_full = current_portofolio.dropna(axis=1, how="all")
            if not sample_for_real_portofolio:
                current_portofolio = current_portofolio.loc[:buy_date]
            else:
                current_portofolio = current_portofolio.loc[buy_date:]

            # dropnan
            current_portofolio.dropna(
                axis=1,
                how="all",
                inplace=True
                )
            if len(current_portofolio) < 10:
                # remove the tickers from all_stock
                continue

            try:
                #mu = expected_returns.mean_historical_return(prices=current_portofolio)
                S = risk_models.risk_matrix(prices=current_portofolio, method="ledoit_wolf")
                #weights = ef.clean_weights()

                # same price for all stocks
                delta = black_litterman.market_implied_risk_aversion(current_portofolio)
                if len(delta) < 10:
                    continue
                viewdict = {
                    current_portofolio.columns[0]: delta[current_portofolio.columns[0]],
                    current_portofolio.columns[1]: delta[current_portofolio.columns[1]],
                    current_portofolio.columns[2]: delta[current_portofolio.columns[2]],
                    current_portofolio.columns[3]: delta[current_portofolio.columns[3]],
                    current_portofolio.columns[4]: delta[current_portofolio.columns[4]],
                    current_portofolio.columns[5]: delta[current_portofolio.columns[5]],
                    current_portofolio.columns[6]: delta[current_portofolio.columns[6]],
                    current_portofolio.columns[7]: delta[current_portofolio.columns[7]],
                    current_portofolio.columns[8]: delta[current_portofolio.columns[8]],
                    current_portofolio.columns[9]: delta[current_portofolio.columns[9]],
                }
                bl = BlackLittermanModel(cov_matrix=S, absolute_views=viewdict)
                mu = bl.bl_returns()
                S = bl.bl_cov()
                bl.bl_weights()
                weights = bl.clean_weights()
                bl.set_weights(weights)
                #expected_returns, volatility, sharpe = bl.portfolio_performance(verbose=False)

                ef = EfficientFrontier(mu, S, weight_bounds=[0.05, 1])
                ef.add_objective(objective_functions.L2_reg, gamma=1)
                ef.min_volatility()
                weights = ef.clean_weights()
                expected_returns, volatility, sharpe = ef.portfolio_performance(verbose=False)

            except Exception as e:
                print(e)
                continue

            if sharpe > 1.5 and sharpe < 10.5:
                annual_return_best = expected_returns
                annual_volatility_best = volatility
                sharpe_best = sharpe
                best_portofolio_weights = weights
                best_portofolio = current_portofolio

                # ('STOCK.OL', WEIGHTs)
                weights_dic = {k: v for k, v in weights.items()}
                # into df
                weights_df = pd.DataFrame(weights_dic, index=[0])

                num_of_stocks_to_buy = (cash / best_portofolio.iloc[-1]) * weights_df
                num_to_buy = round(num_of_stocks_to_buy)
                earned_money = pf.simulate_portofolio_ahead(
                    all_stock=all_stock,
                    weights=weights,
                    input_credits=cash,
                    buy_date=buy_date,
                    show_plot=False
                    )
                if earned_money > 0.07 * cash * total_years:
                    num_of_fail_and_success["success"] += 1
                else:
                    num_of_fail_and_success["fail"] += 1
                earned_money_all.append(earned_money)

                if i % 100 == 0:
                    # plot the weighted portofolio time series
                    diff = current_portofolio_full.pct_change()

                    diff = (diff * best_portofolio_weights).sum(axis=1)

                    # cumsum
                    diff = (1 + diff).cumprod() - 1
                    diff.plot()
                    # red line at buy date
                    plt.axvline(x=buy_date, color="r", linestyle="--")
                    plt.title(f"sharp: {sharpe_best:.2f} | Expected return: {annual_return_best:.2%} | std: {annual_volatility_best:.2%}")
                    mplcyberpunk.add_glow_effects()
                    plt.savefig(f"./examples/weighted_portofolio_{sharpe_best}.png")
                    plt.close()
                    
                    with open(f"./examples/weighted_portofolio_{sharpe_best}.txt", "w") as f:
                        f.write(f'Best portofolio:\n{best_portofolio_weights}\n')
                        f.write(f'Annual return:\n {annual_return_best:.2%}, Annual volatility: {annual_volatility_best:.2%}, Sharpe Ratio: {sharpe_best:.2f}\n')
                        f.write(f'\nNumber of stocks to buy:\n{num_to_buy}')
                        f.write(f'\nEarned money:\n{earned_money}')

        print(f"N = {len(earned_money_all)} Earned: μ={round(np.mean(earned_money_all))}, σ={round(np.std(earned_money_all))} total % increase: {round(np.mean(earned_money_all))/ cash * 100}) (+- {round(np.std(earned_money_all) / cash * 100)}) % over {total_years} years")
        print(f"Number of success: {num_of_fail_and_success['success']}")
        print(f"Number of fail: {num_of_fail_and_success['fail']}")
        print(f"Biggest drawdown: {round(np.min(earned_money_all) / cash * 100)} %")
        print(f"Chance of success {num_of_fail_and_success['success'] / (num_of_fail_and_success['success'] + num_of_fail_and_success['fail']) * 100} %")