from datetime import timedelta
from math import floor

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from scipy import stats
from sklearn import linear_model


class PairsTradingAlgorithm(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2009, 1, 1)
        self.SetEndDate(2017, 1, 1)
        self.SetCash(10000)
        self.numdays = 250  # set the length of training period
        self.tickers = ["XOM", "CVX"]
        self.symbols = []

        self.threshold = 1.
        for i in self.tickers:
            self.symbols.append(self.AddSecurity(
                SecurityType.Equity, i, Resolution.Daily).Symbol)
        for i in self.symbols:
            i.hist_window = RollingWindow[TradeBar](self.numdays)

    def OnData(self, data):

        if not (data.ContainsKey(self.tickers[1]) and data.ContainsKey(self.tickers[0])):
            return

        for symbol in self.symbols:
            symbol.hist_window.Add(data[symbol])

        price_x = pd.Series([float(i.Close) for i in self.symbols[0].hist_window],
                            index=[i.Time for i in self.symbols[0].hist_window])

        price_y = pd.Series([float(i.Close) for i in self.symbols[1].hist_window],
                            index=[i.Time for i in self.symbols[1].hist_window])
        if len(price_x) < 250:
            return
        spread = self.regr(np.log(price_x), np.log(price_y))
        mean = np.mean(spread)
        std = np.std(spread)
        ratio = floor(
            self.Portfolio[self.symbols[1]].Price / self.Portfolio[self.symbols[0]].Price)
        # quantity = float(self.CalculateOrderQuantity(self.symbols[0],0.4))

        if spread[-1] > mean + self.threshold * std:
            if not self.Portfolio[self.symbols[0]].Quantity > 0 and not self.Portfolio[self.symbols[0]].Quantity < 0:
                self.Sell(self.symbols[1], 100)
                self.Buy(self.symbols[0],  ratio * 100)

        elif spread[-1] < mean - self.threshold * std:
            if not self.Portfolio[self.symbols[0]].Quantity < 0 and not self.Portfolio[self.symbols[0]].Quantity > 0:
                self.Sell(self.symbols[0], 100)
                self.Buy(self.symbols[1], ratio * 100)

        else:
            self.Liquidate()

    # build in kalman filter here
    # def regr(self,x,y):
    #     regr = linear_model.LinearRegression()
    #     x_constant = np.column_stack([np.ones(len(x)), x])
    #     regr.fit(x_constant, y)
    #     beta = regr.coef_[0]
    #     alpha = regr.intercept_
    #     spread = y - x*beta - alpha
    #     return spread

    def regr(self, x, y):
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2)

        obs_mat_F = np.transpose(np.vstack(
            [x, np.ones(x.shape[0])])).reshape(-1, 1, 2)

        kf = KalmanFilter(n_dim_obs=1,
                          n_dim_state=2,
                          initial_state_mean=np.ones(2),
                          initial_state_covariance=np.ones((2, 2)),
                          transition_matrices=trans_cov,
                          observation_matrices=obs_mat_F,
                          observation_covariance=1,
                          transition_covariance=np.eye(2))

        state_means, = kf.filter(y)

        slope = state_means[-1, 0]
        intercept = state_means[-1, 1]

        spread = y - x*slope - intercept
        return spread
