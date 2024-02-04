# Load packages
import numpy as np
import pandas as pd

# BaselineModel is created for testing purpose, not meant to be used

class BaselineModel:
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
        # in the horizon, if price at t is higher than 75% percentile, then discharge
        # vice versa, if price is lower than 25% percentile, then charge

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

            if price_indicators_at_t['Market_buy_signal']:
                remaining_volume = self.V_H - S
                frequency = self.trading_freq[price_indicators_at_t['lowest_price_market']-1]

                if t%frequency == 0:
                    charge = min(self.K_C, (remaining_volume/self.eta_C/self.delta_t)/frequency)
                    charge_col_name = f"Charge (MW) from market {price_indicators_at_t['lowest_price_market']}"
                    price_indicators.iloc[t:t+frequency, price_indicators.columns.get_loc(charge_col_name)] = charge
                    
                    dSdt = charge * self.delta_t * self.eta_C
                    charge_profile = S + np.cumsum(np.ones(frequency) * dSdt)
                    price_indicators.iloc[t:t+frequency, price_indicators.columns.get_loc('Storage value (MWh)')] = charge_profile

                    S = price_indicators['Storage value (MWh)'].iloc[t+frequency-1]
                    t += frequency
                else:
                    t += 1

            elif price_indicators_at_t['Market_sell_signal']:
                available_volume = S 
                frequency = self.trading_freq[price_indicators_at_t['highest_price_market']-1]
                
                if t%frequency == 0:
                    discharge = min(self.K_D, (available_volume*self.eta_D/self.delta_t)/frequency)
                    discharge_col_name = f"Discharge (MW) from market {price_indicators_at_t['lowest_price_market']}"
                    price_indicators.iloc[t:t+frequency, price_indicators.columns.get_loc(discharge_col_name)] = discharge

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
        output = self.get_output()

        # Compute profit gained at each time
        profit = 0
        for m in self.markets:
            profit += (output[f'Market {m} Price [£/MWh]'] * output[f'Charge (MW) from market {m}'] * self.delta_t).sum()
            profit -= (output[f'Market {m} Price [£/MWh]'] * output[f'Discharge (MW) from market {m}'] * self.delta_t).sum()
        
        return profit