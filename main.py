import itertools as it
from collections import deque
from datetime import timedelta
from decimal import Decimal
from math import floor

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter
from QuantConnect import Resolution, Symbol
from QuantConnect.Algorithm import QCAlgorithm
from scipy import stats


class PairsTrading(QCAlgorithm):

    def Initialize(self) -> None:

        self.tickers = ['HFC', 'PSX']
        # backtest period to be changed
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2020, 12, 31)

        self.SetCash(10000)

        daily_tickers = ['SPY']
        daily_tickers = daily_tickers.append(self.tickers)
        for i in daily_tickers:
            self.AddEquity(i, Resolution.Daily)
        # calculating beta off previous 30 day window
        self.thirtyDay = self.History(
            [self.Symbol("i") for i in daily_tickers], 30)

        # Minute data
        self.symbols = []
        for i in self.tickers:
            self.symbols.append(self.AddEquity(i, Resolution.Minute).Symbol)

        # holds close price for each ticker
        self.Data = {}
        self.Data[str(self.symbols[0])] = []
        self.Data[str(self.symbols[1])] = []

        # every day set a 30 minute warmup period
        self.Schedule.On(self.DateRules.EveryDay("SPY"),
                         self.TimeRules.AfterMarketOpen("SPY"),
                         self.SetWarmUp(30, Resolution.Minute))

        # Kalman Filter Initialisation
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2)

        self.kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, initial_state_mean=np.zeros(2),
                               initial_state_covariance=np.ones((2, 2)), transition_matrices=np.eye(2),
                               observation_covariance=1.0, transition_covariance=trans_cov)
        self.state_mean = 0.0
        self.state_covariance = 0.0

    def BetaCalculation(self, data):
        """
        A method calculating 30 day beta using daily closing prices
        """
        # assume risk free rate approx 0
        thirtyDayData = self.thirtyDay["close"].unstack(level=0)
        returns = thirtyDayData.pct_change(1)
        cleanedReturns = returns.dropna(axis=0)

        X = cleanedReturns['SPY']
        betas = {}
        for i in self.tickers:
            Y = cleanedReturns[i]
            X1 = sm.add_constant(X)
            model = sm.OLS(Y, X1)
            results = model.fit()
            betas[i] = results.params[1]

        return betas

    def OnData(self, data):
        # adds tick to self.Data
        for symbol in self.symbols:
            if data.Bars.ContainsKey(symbol) and str(symbol) in self.history_price:
                self.Data[str(symbol)].append(float(data[symbol].Close))

        # kalman filter processing
        obs_mat = np.array([[self.Data[0][-1], 1]])

        # input - previous state, previous covariance, new data point, new observation matrix
        # output - new 2D state estimate (slope, intercept), new covariance estimate
        self.state_mean, self.state_covariance = self.kf.filter_update(self.state_mean, self.state_covariance,
                                                                       observation=self.Data[1][-1],
                                                                       observation_matrix=obs_mat)

        if self.IsWarmingUp:

            return
        # test for divergence
            # sell?
            # buy?
        pass
