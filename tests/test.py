import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.battery_model import BatteryModel

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
        
    def output(self):
        # select the market with the 
        pass

class TestBatteryModel(unittest.TestCase):
    # def __init__(self, methodName='runTest', my_value=None):
    #     super().__init__(methodName)
    #     self.my_value = my_value

    def setUp(self):
        self.test_data = pd.read_csv('test_data.csv', index_col=0)
        self.trading_freq = [1, 1, 48]
        self.model = BatteryModel(self.test_data.to_numpy(), trading_freq=self.trading_freq)
        self.model.create_model()
        self.model.solve()
        self.output = self.model.get_output()
        self.output.profit = self.model.get_profits()


    def test_outputs_calculations_correct(self):
        charge = np.zeros(len(self.output))
        discharge = np.zeros(len(self.output))
        status = []

        for m in self.model.markets:
            charge += self.output[f'Charge (MW) from market {m}'].to_numpy()
            discharge += self.output[f'Discharge (MW) from market {m}'].to_numpy() 

        status_prev = self.model.S_init
        for i in range(len(charge)):
            status.append(status_prev + charge[i] * self.model.delta_t * self.model.eta_C - discharge[i] * self.model.delta_t / self.model.eta_D)
            status_prev = status[-1]
        status = np.array(status)

        with self.subTest():
            self.assertTrue(np.all(self.output['Total charge (MW)'].to_numpy() == charge))
        with self.subTest():
            self.assertTrue(np.all(self.output['Total discharge (MW)'].to_numpy() == discharge))
        with self.subTest():
            self.assertTrue(np.all(self.output['Storage value (MWh)'].to_numpy() == status))
        

    # def test_charging_upper_constraints_met(self):
    #     with self.subTest():
    #         self.assertLessEqual(self.output['Total charge (MW)'].max(), self.model.K_C)
    #     with self.subTest():
    #         self.assertLessEqual(self.output['Total discharge (MW)'].max(), self.model.K_D)

    
    # def test_charging_lower_constraints_met(self):
    #     with self.subTest():
    #         self.assertGreaterEqual(self.output['Total charge (MW)'].min(), 0)
    #     with self.subTest():
    #         self.assertGreaterEqual(self.output['Total discharge (MW)'].min(), 0)


    # def test_no_simultaneous_charging_and_discharging(self):
    #     for i in range(len(self.output)):
    #         charge_value = self.output.iloc[i]['Total charge (MW)']
    #         discharge_value = self.output.iloc[i]['Total discharge (MW)']
    #         with self.subTest():
    #             self.assertEqual(charge_value + discharge_value, max(charge_value, discharge_value))


    # def test_storage_status_constraints_met(self):
    #     with self.subTest():
    #         self.assertLessEqual(self.output['Storage value (MWh)'].max(), self.model.V_H)
    #     with self.subTest():
    #         self.assertGreaterEqual(self.output['Storage value (MWh)'].max(), 0)


    # def test_trading_frequency_constraints_met(self):
    #     output = self.output.copy()
    #     for i in self.model.markets:
    #         # filter out non zero values and check the block length
    #         mask = self.output['Discharge (MW) from market 1'] > 0
    #         output['group'] = (mask != mask.shift(1)).cumsum()
    #         non_zero_groups = output[mask].groupby('group')['Discharge (MW) from market 1'].count()

    #         with self.subTest():
    #             self.assertTrue((non_zero_groups >= self.trading_freq[i-1]).all())


    # def test_multiple_market_is_better(self):
    #     market_sets = {(1,), (2,), (3,), (1,2), (1,3), (2,3)}

    #     for market_set in market_sets:
    #         freq = [self.trading_freq[i-1] for i in market_set]
    #         testcase_model = BatteryModel(self.test_data.iloc[:, [0, *market_set]].to_numpy(), trading_freq=freq)
    #         testcase_model.create_model()
    #         testcase_model.solve()
    #         profit = testcase_model.get_profits()
    #         with self.subTest():
    #             self.assertLessEqual(profit, self.output.profit)


    # def test_no_depreciation_gives_more_profit(self):
    #     testcase_model = BatteryModel(self.test_data.to_numpy(), trading_freq=self.trading_freq)
    #     testcase_model.create_model(consider_depreciation=False)
    #     testcase_model.solve()
    #     profit = testcase_model.get_profits()
    #     self.assertLess(self.output.profit, profit)

    
    # def test_no_charging_loss_gives_more_profit(self):
    #     testcase_model = BatteryModel(self.test_data.to_numpy(), trading_freq=self.trading_freq, eta_C=1.0, eta_D=1.0)
    #     testcase_model.create_model()
    #     testcase_model.solve()
    #     profit = testcase_model.get_profits()
    #     self.assertLess(self.output.profit, profit)



if __name__ == '__main__':
    unittest.main(verbosity=2)