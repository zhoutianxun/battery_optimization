#Load packages
import numpy as np
import pandas as pd
import pulp
import matplotlib.pyplot as plt

class battery_model:
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
    trading_freq : iterable, len=number of markets
        the trading frequency of each market in units of time 
        for e.g. if market 1 trades once every half hour, and market 2 trades once every day
        the trading_freq = [1, 48], where each unit of time is half hour
    delta_t : float
        each unit of time in hours
        for e.g. if each unit of time is half hour, delta_t=0.5
    horizon_length : int
        the planning horizon (in unit of time) for which to run the model
        for e.g. if horizon is 48 hours, and delta_t=0.5, horizon_length=96
    times : range
        range of time for the horizon, =range(horizon_length)
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
    model_created : bool
        flag for if model has been created via create_model() method
    model_solved : bool
        flag for if model has been solved via solve() method
    
    Methods
    -------
    create_model():
        initializes PuLP linear programming model with control variables, objective function, and constraints
    solve():
        applies solver to solve linear programming problem and obtain optimal values for control variables
    get_output():
        returns an Pandas dataframe that organizes the output of model into tabular form
    get_profits(until=None):
        returns profit gained by applying plan up to time of 'until'
    plot_output(prices, output, figsize=(6,12)):
        generates a Matplotlib Pyplot of the output of the model for visualization
    """

    def __init__(self, 
                 prices, 
                 trading_freq=None, 
                 delta_t=0.5,
                 horizon_length=144, 
                 S_init=0,
                 V_H=4,
                 n_H=0,
                 n_max=5000,
                 gamma_H=500000,
                 K_C=2,
                 K_D=2,
                 eta_C=0.95,
                 eta_D=0.95,
                 f=1e-5):
        
        """
        Constructs all the necessary attributes for the battery_model object

        Parameters
        ----------
            prices : numpy.ndarray, shape=(length of time horizon, number of markets)
                the price data of electricity (Pound per MW) in the planning horizon for all the markets
            trading_freq : iterable, optional (default=None), len=number of markets
                the trading frequency of each market in units of time 
                for e.g. if market 1 trades once every half hour, and market 2 trades once every day
                the trading_freq = [1, 48], where each unit of time is half hour
                if None is given, defaults to value of 1 for every market
            delta_t : float, optional (default=0.5)
                each unit of time in hours
                for e.g. if each unit of time is half hour, delta_t=0.5
            horizon_length : int, optional (default=144)
                the planning horizon (in unit of time) for which to run the model
                for e.g. if horizon is 48 hours, and delta_t=0.5, horizon_length=96
            S_init : float, optional (default=0), value between 0 to V_H
                the initial charge status of the battery in MWh
            V_H : float, optional (default=4)
                the volume of the battery at the start of this horizon in MWh
            n_H : float, optional (default=0)
                the number of charging cycle completed at the start of this horizon
            n_max : float, optional (default=5000)
                the max number of charging cycle for the battery
            gamma_H : float, optional (default=500000)
                the remaining (capital) value of the battery in Pound
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
        if trading_freq == None:
            self.trading_freq = [1] * (prices.shape[1]-1)
        else:
            assert hasattr(trading_freq, '__iter__') and len(trading_freq) == prices.shape[1]-1, \
            "trading_freq needs to be an interable with same length as number of markets as \
                provided in prices"
            self.trading_freq = trading_freq
        self.delta_t = delta_t
        self.horizon_length = horizon_length
        self.times = range(horizon_length)
        assert S_init <= V_H
        self.S_init = S_init
        assert V_H > 0, "battery volume is exhausted"
        assert n_max - n_H > 0, "battery has no remaining life as max cycle is reached"
        assert gamma_H > 0, "battery has no remaining life as max lifetime is reached"
        self.V_H = V_H
        self.depreciation_cost = gamma_H/((n_max - n_H) * V_H)
        self.K_C= K_C
        self.K_D= K_D
        self.eta_C= eta_C
        self.eta_D= eta_D
        self.f = f

        self.model_created = False
        self.solved = False


    def __str__(self) -> str:
        """
        Returns
        -------
        str : string representation of object
            if model has been created with create_model() method, return the PuLP model string representation
            which contains full details of the linear program
        """

        if self.model_created == True:
            return self.prob.__str__()
        else:
            return "Uninitated battery_model"


    def create_model(self) -> None:
        """
        initializes PuLP linear programming model with control variables, objective function, and constraints

        the mixed-integer program consists of 3 parts
        1. CONTROL VARIABLES
            the two main sets of control variables are Charge and Discharge values for every timestep in the horizon
            the third set is a binary control variable, Mode indicate the mode of operation (either charging or discharging)
            the Mode variables are used to enforce the constraint that charging and discharging cannot take place simultaneously
        
        2. OBJECTIVE FUNCTION
            the objective function is profit which is to be maximized
            the profit function is defined as revenue - cost, which further breaks down into
            profit = discharging revenue - charging cost - depreciation cost
            discharging revenue is the Charge value * market price 
            charging cost is the Discharge value * market price
            depreciation cost is the amount of charge cycle * capital depreciation cost per charge cycle

        3. CONSTRAINTS
            there are 3 main sets of constraints:
            i) charging and discharging does not exceed max charging/discharging capacity, 
               and charging and discharging does not occur simultaneously
               this is implemented through Big-M constraint method to set sum of charging or discharging binary Mode variable to be <=1
               and using Big-M value of max charging and discharging rate as the upper bound for total Charge and Discharge value
            
            ii) storage status of battery does not exceed battery volume
               the lower bound is 0, and the upper bound is battery volume
               storage status can be expressed in terms of S_init, cumulative Charge and Discharge values
               battery volume is a changing variable as it decreases with charging due to degradation
               an approximation of battery volume as a function of cumulative Charge value is used as the upper bound
            
            iii) trading in market needs to follow market trading frequency
               based on trading frequency of a market, the battery must export/import a constant level of power for the full period
               e.g. if the market trades on daily frequency, the battery can only import/export a constant level of power for the full day
               this is implemented by constraining decision variables at mod(time, trading frequency) =/= 0 
               to follow the decision variable value at mod(time, trading frequency) = 0
        
        Returns
        -------
        None
        """
                
        # initiate problem
        prob = pulp.LpProblem("battery_model", pulp.LpMaximize)

        # add control variable
        self._add_variables()

        # add objective function
        prob = self._add_objective(prob)

        # add constraints
        prob = self._add_capacity_constraints(prob)
        prob = self._add_volume_constraints(prob)
        prob = self._add_frequency_constraints(prob)

        # update attribute and flag
        self.prob = prob
        self.model_created = True

    def _add_variables(self) -> None:
        """
        Add control variables:
            Charge : Continuous
                charge values for all timesteps and for all markets
            Discharge : Continuous
                Continuousdischarge values for all timesteps and for all markets
            Mode : Binary
                Continuous mode of operation all timesteps and for either charging ('c') or discharing ('d')

        Returns
        -------
        None
        """
        self.charge = pulp.LpVariable.dicts("Charge", 
                                            [f'C_m{m}_t{t}' for t in self.times for m in self.markets], 
                                            lowBound=0, 
                                            upBound=self.K_C, 
                                            cat='Continuous')
        
        self.discharge = pulp.LpVariable.dicts("Discharge", 
                                               [f'D_m{m}_t{t}' for t in self.times for m in self.markets], 
                                               lowBound=0, 
                                               upBound=self.K_D, 
                                               cat='Continuous')
                    
        self.mode = pulp.LpVariable.dicts("Mode", 
                                          [f'M_{op}_t{t}' for t in self.times for op in ('c', 'd')], 
                                          lowBound=0, 
                                          upBound=1, 
                                          cat='Binary')
    

    def _add_objective(self, prob) -> pulp.LpProblem:
        """
        Add objective function
            profit 
            = discharging revenue
                Discharge value (MW) * discharge time (h) * electricity price (Pound/MWh) -> Pound
            - charging cost
                Charge value (MW) * charge time (h) * electricity price (Pound/MWh) -> Pound
            - depreciation cost
                Charge value (MW) * charge time (h) * depreciation cost (Pound/MWh) -> Pound

        Parameters
        ----------
        prob : pulp.LpProblem
            LP problem without objective function

        Returns
        -------
        prob : pulp.LpProblem
            LP problem with objective function added
        """

        prob += (
              pulp.LpAffineExpression(
                  [(self.discharge[f'D_m{m}_t{t}'], self.delta_t*self.prices[t, m]) 
                   for t in self.times for m in self.markets])
            - pulp.LpAffineExpression(
                  [(self.charge[f'C_m{m}_t{t}'], self.delta_t*self.prices[t, m]) 
                   for t in self.times for m in self.markets])
            - pulp.LpAffineExpression(
                  [(self.charge[f'C_m{m}_t{t}'], self.delta_t*self.depreciation_cost) 
                   for t in self.times for m in self.markets])
        )
        return prob
    

    def _add_capacity_constraints(self, prob):
        """
        Add capacity constraint
            Binary Mode variable for charging ('c') and discharging ('d') at any t
                sum of modes <= 1 to enforce no simultaneous charging rule
            Charge variable at any t
                sum over all markets <= max charging capacity, if Mode is charging
            Discharge variable at any t
                sum over all markets <= max discharging capacity, if Mode is discharging

        Parameters
        ----------
        prob : pulp.LpProblem
            LP problem without capacity constraint

        Returns
        -------
        prob : pulp.LpProblem
            LP problem with capacity constraint added
        """
                
        for t in self.times:
            prob += (
                pulp.lpSum([self.mode[f'M_{op}_t{t}'] for op in ('c', 'd')]) 
                <= 1
                )
            prob += (
                pulp.lpSum([self.charge[f'C_m{m}_t{t}'] for m in self.markets]) 
                <= self.K_C * self.mode[f'M_c_t{t}']
                )
            prob += (
                pulp.lpSum([self.discharge[f'D_m{m}_t{t}'] for m in self.markets]) 
                <= self.K_D * self.mode[f'M_d_t{t}']
                )
        return prob
    

    def _add_volume_constraints(self, prob):
        # 2. storage status of battery does not exceed battery volume
        #    storage constraint, 0 <= S_t <= Capacity

        """
        Add volume constraint
            Charge status at any t as a function of cumulative Charge and Discharge value
            S_t = S_0 (MWh) + delta_t (h) * eta_C * cumsum(C_t) (MW) - delta_t (h) * 1/eta_D * cumsum(D_t) (MW) -> MWh
            V_t = V_0 - f * cumsum(C_t) 
            
            lower bound on charge status
                charge status (S_t) >= 0 for all t
            upper bound on charge status
                charge status - battery volume (V_t) <= 0 at t for all t

        Parameters
        ----------
        prob : pulp.LpProblem
            LP problem without volume constraint

        Returns
        -------
        prob : pulp.LpProblem
            LP problem with volume constraint added
        """
                
        for t in self.times:
            # lower bound
            prob += (
                  self.S_init
                + pulp.LpAffineExpression(
                    [(self.charge[f'C_m{m}_t{i}'], self.delta_t * self.eta_C) 
                     for i in range(t+1) for m in self.markets])
                - pulp.LpAffineExpression(
                    [(self.discharge[f'D_m{m}_t{i}'], self.delta_t * 1/self.eta_D) 
                     for i in range(t+1) for m in self.markets])
                >= 0
            )
    
            # upper bound   
            prob += (
                  self.S_init
                - self.V_H
                + pulp.LpAffineExpression(
                    [(self.charge[f'C_m{m}_t{i}'], self.delta_t * (self.eta_C + self.f)) 
                     for i in range(t+1) for m in self.markets])
                - pulp.LpAffineExpression(
                    [(self.discharge[f'D_m{m}_t{i}'], self.delta_t * 1/self.eta_D) 
                     for i in range(t+1) for m in self.markets])
                <= 0
            )
        return prob
    
    
    def _add_frequency_constraints(self, prob):
        """
        Add frequency constraint
            trading in market needs to follow market trading frequency
            at time (t % trading frequency) =/= 0, charging and discharging for market
            proceeds at rate set at (t % trading frequency) == 0
                
            Charge variable (C_t) at any (t % trading frequency) =/= 0
                C_t = C_{t-1}
            Discharge variable (D_t) at any (t % trading frequency) =/= 0
                D_t = D_{t-1}

        Parameters
        ----------
        prob : pulp.LpProblem
            LP problem without frequency constraint

        Returns
        -------
        prob : pulp.LpProblem
            LP problem with frequency constraint added
        """
                
        for m in self.markets:
            for t in self.times:
                if t % self.trading_freq[m-1] != 0:
                    prob += self.charge[f'C_m{m}_t{t}'] - self.charge[f'C_m{m}_t{t-1}'] == 0 
                    prob += self.discharge[f'D_m{m}_t{t}'] - self.discharge[f'D_m{m}_t{t-1}'] == 0 
        
        return prob
    

    def solve(self) -> None:
        """
        applies solver to solve linear programming problem and obtain optimal values for control variables

        Returns
        -------
        None
        """

        if self.model_created == False:
            raise Exception("Model not created yet, run create_model method first")
        
        self.prob.solve(pulp.apis.PULP_CBC_CMD(msg=0))
        self.solved = True


    def get_output(self) -> pd.DataFrame:
        """
        returns an Pandas dataframe that organizes the output of model into tabular form

        Returns
        -------
        output : pd.DataFrame
            contains columns 'Datetime', 'Charge (MW) from market {m}', 'Discharge (MW) from market {m}',
            'Total charge (MW)', 'Total discharge (MW)', 'Storage value (MWh)', 'Profits (£)'
        """

        if self.solved == False:
            raise Exception("Model not solved yet, run solve method first")
        
        # Gather charge and discharge values at each time
        charge_values = np.zeros((len(self.markets), len(self.times)))
        discharge_values = np.zeros((len(self.markets), len(self.times)))
        
        for m in self.markets:
            charge_values[m-1] = np.array([self.charge[f'C_m{m}_t{t}'].varValue for t in self.times])
            discharge_values[m-1] = np.array([self.discharge[f'D_m{m}_t{t}'].varValue for t in self.times])
        
        # Compute profit gained at each time
        profit = (
            discharge_values * self.prices[:, 1:].T * self.delta_t
            - charge_values * self.prices[:, 1:].T * self.delta_t
            )
        profit = np.sum(profit, axis=0)

        # Compute amount stored in battery at end of each time
        storage_values = []
        S_t = self.S_init
        for t in self.times:
            S_t += (self.eta_C * np.sum(charge_values[:, t]) 
                    - 1/self.eta_D * np.sum(discharge_values[:, t])) * self.delta_t
            storage_values.append(S_t)

        output = pd.DataFrame({'Datetime': self.prices[:, 0]})
        for m in self.markets:
            output[f'Charge (MW) from market {m}'] = charge_values[m-1]
            output[f'Discharge (MW) from market {m}'] = discharge_values[m-1]
        output['Total charge (MW)'] = np.sum(charge_values, axis=0)
        output['Total discharge (MW)'] = np.sum(discharge_values, axis=0)
        output['Storage value (MWh)'] = storage_values
        output['Profits (£)'] = profit

        return output


    def get_profits(self, until=None) -> float:
        """
        returns profit gained by applying plan up to time of 'until'
        e.g. if horizon_length analyzed is 144, but plan is only executed up to t=48,
        then the actual profit realized up to t=48 can be found by get_profits(until=48)
        
        If the argument 'until' is not passed, then method returns profit until the end of horizon given 

        Parameters
        ----------
        until : str, optional (default=None)
            compute profit from start of horizon up to t='until'
            if None, method returns profit until the end of horizon given
        
        Returns
        -------
        float : 
            profit of time period specified. Note that profit does not consider depreciation cost (aka EBITDA)
        """

        if self.solved == False:
            raise Exception("Model not solved yet, run solve method first")  

        output = self.get_output()
        if until == None:
            until = self.horizon_length
        else:
            assert type(until) == int, "until needs to be an integer"

        return sum(output['Profits (£)'].iloc[:until])
    

    @staticmethod
    def plot_output(prices, output, figsize=(6, 12)) -> None:
        """
        generates a Matplotlib Pyplot of the output of the model for visualization

        Parameters
        ----------
        prices : pd.DataFrame or np.ndarray, shape=(length of time, number of markets)
            the price data of electricity (Pound per MW) for all the markets
        output : pd.DataFrame, same length as prices
            output dataframe containing information required for plotting
            obtained from get_output() method
        figsize : tuple, optional (default=(6, 12))
            controls figure size of plot
        
        Returns
        -------
        None
        """

        assert len(prices) == len(output), "lengths of 'prices' and 'output' are different" 
        if type(prices) == pd.DataFrame:
            prices = prices.to_numpy()

        markets = range(1, prices.shape[1])
        charge_values = np.zeros((len(markets), len(output)))
        discharge_values = np.zeros((len(markets), len(output)))
        for m in markets:
            charge_values[m-1] = output[f'Charge (MW) from market {m}']
            discharge_values[m-1] = output[f'Discharge (MW) from market {m}']
        total_charge = output['Total charge (MW)']
        total_discharge = output['Total discharge (MW)']
        storage_values = output['Storage value (MWh)']
        profit = output['Profits (£)'].cumsum()

        _, axs = plt.subplots(5, 1, figsize=figsize)
        axs[0].set_title("Market prices")
        axs[0].set_ylabel("£/MW")
        axs[1].set_title("Charging and discharging signals (overall)")
        axs[1].set_ylabel("MW")
        axs[2].set_title("Charging and discharging signals for each market")
        axs[2].set_ylabel("MW")
        axs[3].set_title("Battery storage")
        axs[3].set_ylabel("MWh")
        axs[4].set_title("Cumulative profit")
        axs[4].set_ylabel("£")
        axs[4].set_xlabel("time units")

        for m in markets:
            axs[0].plot(np.arange(len(output)), prices[:, m], label=f'market {m}')
            axs[2].plot(np.arange(len(output)), discharge_values[m-1], label=f'discharge_{m}')
            axs[2].plot(np.arange(len(output)), charge_values[m-1], label=f'charge_{m}')
        
        axs[1].plot(np.arange(len(output)), total_charge, label='total charge')
        axs[1].plot(np.arange(len(output)), total_discharge, label='total discharge')
        axs[3].plot(np.arange(len(output)), storage_values, label='storage')
        axs[4].plot(np.arange(len(output)), profit, label='cumulative profit')
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        plt.tight_layout()
            