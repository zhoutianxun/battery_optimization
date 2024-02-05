# Load packages
import numpy as np
import pandas as pd

# BaselineModel is created for testing purpose, not meant to be used

class BaselineModel:
    """
    Linear programming optimisation model to charge/discharge the battery over a time horizon in order to maximise profits
    The model finds the optimal trade at each time across several electricity markets
    ...

    Attributes
    ----------
    prices : numpy.ndarray, shape=(length of time horizon, number of markets)
        the price data of electricity (Pound per MW) in the planning horizon for all the markets
    markets : range
        list of markets, generated automatically from inferring number of markets from prices table
        e.g. if there are 3 market prices provided, markets=range(1, 3+1)
    delta_t : float
        each unit of time in hours
        for e.g. if each unit of time is half hour, delta_t=0.5
    trading_freq : iterable, len=number of markets
        the trading frequency of each market in units of time 
        for e.g. if market 1 trades once every half hour, and market 2 trades once every day
        the trading_freq = [1, 48], where each unit of time is half hour
    horizon_length : int
        the planning horizon (in unit of time) for which to run the model
        for e.g. if horizon is 48 hours, and delta_t=0.5, horizon_length=96
    S_init : float, value between 0 to V_H
        the initial charge status of the battery in MWh
    V_H : float
        the volume of the battery at the start of this horizon in MWh
    depreciation_cost : float
        the depreciation cost incurred per unit (MW) of charge
        defined as the remaining (capital) value of the battery in Pounds divided by remaining charge cycles
    K_C : float
        the maximum charging rate of the battery in MW
    K_D : float
        the maximum discharging rate of the battery in MW
    eta_C : float
        the battery charging efficiency
        for e.g. when eta_C=0.95, charging at the rate of 1 MW for 1 hour, the battery gains 0.95 MWh
    eta_D : float
        the battery discharging efficiency
        for e.g. when eta_D=0.95, discharging at the rate of 0.95 MW for 1 hour, the battery loses 1 MWh
    f : float
        battery degradation per cycle
        for e.g. after 1 cycle of charging, battery volume becomes (1-f)*previous volume
    
    Methods
    -------
    get_output():
        runs the baseline model and returns an Pandas dataframe that organizes the output of model into tabular form
    """

    def __init__(self, 
                 prices, 
                 delta_t=0.5,
                 trading_freq=[1, 1, 48],
                 horizon_length=144, 
                 S_init=0,
                 V_H=4,
                 K_C=2,
                 K_D=2,
                 eta_C=0.95,
                 eta_D=0.95,
                 f=1e-5):
        
        """
        Constructs all the necessary attributes for the baseline_model object

        Parameters
        ----------
            prices : numpy.ndarray, shape=(length of time horizon, number of markets)
                the price data of electricity (Pound per MW) in the planning horizon for all the markets
            delta_t : float, optional (default=0.5)
                each unit of time in hours
                for e.g. if each unit of time is half hour, delta_t=0.5
            trading_freq : iterable, optional (default=None), len=number of markets
                the trading frequency of each market in units of time 
                for e.g. if market 1 trades once every half hour, and market 2 trades once every day
                the trading_freq = [1, 48], where each unit of time is half hour
                if None is given, defaults to value of 1 for every market
            horizon_length : int, optional (default=144)
                the planning horizon (in unit of time) for which to run the model
                for e.g. if horizon is 48 hours, and delta_t=0.5, horizon_length=96
            S_init : float, optional (default=0), value between 0 to V_H
                the initial charge status of the battery in MWh
            V_H : float, optional (default=4)
                the volume of the battery at the start of this horizon in MWh
            K_C : float, optional (default=2)
                the maximum charging rate of the battery in MW
            K_D : float, optional (default=2)
                the maximum discharging rate of the battery in MW
            eta_C : float, optional (default=0.95)
                the battery charging efficiency
                for e.g. when eta_C=0.95, charging at the rate of 1 MW for 1 hour, the battery gains 0.95 MWh
            eta_D : float, optional (default=0.95)
                the battery discharging efficiency
                for e.g. when eta_D=0.95, discharging at the rate of 0.95 MW for 1 hour, the battery loses 1 MWh
            f : float, optional (default=1e-5)
                battery degradation per cycle
                for e.g. after 1 cycle of charging, battery volume becomes (1-f)*previous volume
        
        Returns
        -------
        None
        """

        self.prices = prices
        self.markets = range(1, prices.shape[1])
        self.delta_t = delta_t
        self.trading_freq = trading_freq
        self.horizon_length = horizon_length
        self.S_init = S_init
        self.V_H = V_H
        self.K_C = K_C
        self.K_D = K_D
        self.eta_C = eta_C
        self.eta_D = eta_D
        self.f = f


    def get_output(self):
        """
        runs a simple baseline method for charge and discharge planning.
        
        At each given time in the horizon, look at the highest and lowest price among all markets
        if the highest price at t is higher than 75% percentile, then discharge
        vice versa, if the lowest price is lower than 25% percentile, then charge
        
        returns an Pandas dataframe that organizes the output of model into tabular form

        Returns
        -------
        output : pd.DataFrame
            contains columns 'Datetime', 'Charge (MW) from market {m}', 'Discharge (MW) from market {m}',
            'Total charge (MW)', 'Total discharge (MW)', 'Storage value (MWh)', 'Profits (£)'
        """

        price_indicators = self.prices.copy()

        # get highest and lowest price among all markets at each time
        price_indicators['highest_price_market'] = price_indicators.iloc[:, 1:1+len(self.markets)].idxmax(axis=1)
        price_indicators['lowest_price_market'] = price_indicators.iloc[:, 1:1+len(self.markets)].idxmax(axis=1)
        price_indicators['highest_price_market'] = price_indicators['highest_price_market'].map(lambda name: int(name.split(' ')[1]))
        price_indicators['lowest_price_market'] = price_indicators['lowest_price_market'].map(lambda name: int(name.split(' ')[1]))

        price_indicators['highest_price'] = price_indicators.iloc[:, 1:1+len(self.markets)].max(axis=1)
        price_indicators['lowest_price'] = price_indicators.iloc[:, 1:1+len(self.markets)].min(axis=1)
        
        price_indicators[f'Market_sell_signal'] = price_indicators['highest_price'] >= price_indicators['highest_price'].quantile(0.75)
        price_indicators[f'Market_buy_signal'] = price_indicators['lowest_price'] <= price_indicators['lowest_price'].quantile(0.25)

        # run strategy
        S = self.S_init
        for m in self.markets:
            price_indicators[f'Charge (MW) from market {m}'] = 0.
            price_indicators[f'Discharge (MW) from market {m}'] = 0.
        price_indicators['Storage value (MWh)'] = 0.

        t = 0
        while t <len(price_indicators):
            price_indicators_at_t = price_indicators.iloc[t]
            price_indicators.iloc[t, price_indicators.columns.get_loc('Storage value (MWh)')] = S

            # if price signal is buy
            if price_indicators_at_t['Market_buy_signal']:
                remaining_volume = self.V_H - S
                frequency = self.trading_freq[price_indicators_at_t['lowest_price_market']-1]

                # enforce trading frequency constraint
                if t%frequency == 0:
                    
                    # compute how much it's possible to charge based on remaining available volume
                    charge = min(self.K_C, (remaining_volume/self.eta_C/self.delta_t)/frequency)
                    charge_col_name = f"Charge (MW) from market {price_indicators_at_t['lowest_price_market']}"
                    price_indicators.iloc[t:t+frequency, price_indicators.columns.get_loc(charge_col_name)] = charge
                    
                    # update storage value
                    dSdt = charge * self.delta_t * self.eta_C
                    charge_profile = S + np.cumsum(np.ones(frequency) * dSdt)
                    price_indicators.iloc[t:t+frequency, price_indicators.columns.get_loc('Storage value (MWh)')] = charge_profile

                    S = price_indicators['Storage value (MWh)'].iloc[t+frequency-1]
                    t += frequency
                else:
                    t += 1

            # if price signal is sell
            elif price_indicators_at_t['Market_sell_signal']:
                available_volume = S 
                frequency = self.trading_freq[price_indicators_at_t['highest_price_market']-1]
                
                # enforce trading frequency constraint
                if t%frequency == 0:

                    # compute how much it's possible to discharge based on available volume
                    discharge = min(self.K_D, (available_volume*self.eta_D/self.delta_t)/frequency)
                    discharge_col_name = f"Discharge (MW) from market {price_indicators_at_t['lowest_price_market']}"
                    price_indicators.iloc[t:t+frequency, price_indicators.columns.get_loc(discharge_col_name)] = discharge

                    # update storage value
                    dSdt = - discharge * self.delta_t / self.eta_D
                    discharge_profile = S + np.cumsum(np.ones(frequency) * dSdt)
                    price_indicators.iloc[t:t+frequency, price_indicators.columns.get_loc('Storage value (MWh)')] = discharge_profile
                    S = price_indicators['Storage value (MWh)'].iloc[t+frequency-1]
                    t += frequency
                else:
                    t += 1

            else:
                t += 1
            
    
        return price_indicators

    def get_profits(self):
        """
        returns profit gained by applying plan
        
        Returns
        -------
        float : 
            profit of time period specified. Note that profit does not consider depreciation cost (aka EBITDA)
        """
                
        output = self.get_output()

        # Compute profit gained at each time
        profit = 0
        for m in self.markets:
            profit += (output[f'Market {m} Price [£/MWh]'] * output[f'Charge (MW) from market {m}'] * self.delta_t).sum()
            profit -= (output[f'Market {m} Price [£/MWh]'] * output[f'Discharge (MW) from market {m}'] * self.delta_t).sum()
        
        return profit