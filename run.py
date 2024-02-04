import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.battery_model import BatteryModel
from src.data_processing import process_data


def run_simulation(battery_specs_dict, 
                   all_data, 
                   planning_horizon=144,
                   execution_horizon=48,  
                   delta_t=0.5,
                   trading_freq=[1, 1, 48], 
                   S_init=0,
                   consider_depreciation=True) -> pd.DataFrame:
    
    """
    run simulation using battery model on a rolling horizon basis
    plan is made using pricing data up till planning horizon
    plan is executed up till execution horizon
    after executing the plan, a new plan is made with the next rolling planning horizon

    Parameters
    ----------
    battery_specs_dict : dict
        dictionary of battery specs 
    all_data : pd.DataFrame
        combined price file for both half hourly and daily electricity market prices
    planning_horizon : int, optional (default=144)
        length of planning horizon, value should be at least 48
    execution_horizon : int, optional (default=48)    
        length of execution horizon, value should be at least 48
        execution horizon should be less than or equal to planning horizon
    delta_t : float, optional (default=0.5)
        each unit of time in hours
        for e.g. if each unit of time is half hour, delta_t=0.5
    trading_freq : iterable, optional (default=[1, 1, 48]), len=number of markets
        the trading frequency of each market in units of time 
        for e.g. if market 1 and 2 trades once every half hour, and market 3 trades once every day
        the trading_freq = [1, 1, 48], where each unit of time is half hour
    S_init : float, optional (default=0), value between 0 to V_H
        the initial charge status of the battery in MWh
    consider_depreciation: bool (default=True)
            does model take depreciation from charging into account
    
    Returns
    -------
    output : pd.DataFrame
        contains columns 'Datetime', 'Charge (MW) from market {m}', 'Discharge (MW) from market {m}',
        'Total charge (MW)', 'Total discharge (MW)', 'Storage value (MWh)', 'Profits (£)',
        'Battery volume (MWh)', 'Battery value (£)', 'Completed cycles'
    """

    assert planning_horizon >= 48, "planning horizon at least 48"
    assert execution_horizon >= 48, "execultion horizon at least 48"
    assert execution_horizon <= planning_horizon, "execution horizon needs to be <= planning horizon"

    START_TIME = all_data.iloc[0, 0]
    V_H = battery_specs_dict['V_init']
    n_H = 0

    # Create dataframe to store outputs
    columns = ['Datetime']
    for m in range(1, all_data.shape[1]):
        columns.append(f'Charge (MW) from market {m}')
        columns.append(f'Discharge (MW) from market {m}')
    columns += ['Total charge (MW)', 'Total discharge (MW)', 'Storage value (MWh)', 'Profits (£)']
    output = pd.DataFrame(columns=columns)

    # Keep track of battery capital value, and completed cycles
    battery_value = np.zeros(len(all_data))
    completed_cycles = np.zeros(len(all_data))
    V_t = np.zeros(len(all_data))

    # Iterate over all time horizons until end
    bar = tqdm(range(0, len(all_data), execution_horizon))
    for H in bar:

        # get planning horizon data
        planning_data = all_data.iloc[H:H+planning_horizon]
        prices = planning_data.to_numpy()

        # compute remaining battery value gamma_H
        horizon_start_time = planning_data.iloc[0, 0]
        time_since_start = horizon_start_time - START_TIME
        gamma_H = (1 - time_since_start.days/(365.25 * 10)) * battery_specs_dict['Capex']

        # Feed data into model and get planning outputs
        model = BatteryModel(prices, 
                             trading_freq=trading_freq, 
                             delta_t=delta_t,
                             horizon_length=len(planning_data), 
                             S_init=S_init,
                             V_H=V_H,
                             n_H=n_H,
                             n_max=battery_specs_dict['n_max'],
                             gamma_H=gamma_H,
                             K_C=battery_specs_dict['K_C'],
                             K_D=battery_specs_dict['K_D'],
                             eta_C=battery_specs_dict['eta_C'],
                             eta_D=battery_specs_dict['eta_D'],
                             f=battery_specs_dict['f'])

        model.create_model(consider_depreciation=consider_depreciation)
        model.solve()

        # Only execute plan up to execution horizon
        plan = model.get_output().iloc[:execution_horizon]
        output = pd.concat((output, plan), ignore_index=True)

        # Compute battery property variables, V_H, n_H for upcoming horizon
        for t in range(H, H+len(plan)):
            if t == 0:
                V_prev = battery_specs_dict['V_init']
            else:
                V_prev = V_t[t-1]
            V_t[t] = (
                V_prev * (1-battery_specs_dict['f'])**(output['Total charge (MW)'].iloc[t]/V_prev)
                )
        
        completed_cycles[H:H+len(plan)] = n_H + (output.iloc[H:H+len(plan)]['Total charge (MW)']/V_t[H:H+len(plan)]).cumsum()
        S_init = output['Storage value (MWh)'].iloc[-1]
        V_H = V_t[t]
        n_H = completed_cycles[t]
        
        # Because we approximated V_t with the value V_H at start of horizon, S_init may exceed actual V_t at end of horizon
        # in such case, we print to screen the error of estimation (%) for S_init, 
        # and force S_init to actual V_t for start of next horizon
        if S_init > V_H:
            S_init = V_H
            bar.set_description(f"error due to approximation of V_t with V_H at horizon t={H}: {(S_init - V_H)/S_init *100:.5f}%")
    
    output['Battery volume (MWh)'] = V_t
    output['Battery value (£)'] = (1 - np.arange(0, len(all_data)*0.5/24, 0.5/24)/(365.25*10)) * battery_specs_dict['Capex']
    output['Completed cycles'] = completed_cycles
    return output

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan_horizon', '-p', help='planning horizon (default=144)', type=int, default=144)
    parser.add_argument('--exec_horizon', '-e', help='execultion horizon (default=48)', type=int, default=48)
    parser.add_argument('--s_init', '-s', help='initial state of charge of battery (default=0)', type=float, default=0)
    parser.add_argument('--out_path', '-o', help='path for saving output (default="output/output.csv")', type=str, default='output/output.csv')
    args = parser.parse_args()

    # Load data
    battery_spec_file = os.path.join('data', 'battery_specs.csv')
    half_hourly_price_file = os.path.join('data', 'half_hourly_data.csv')
    daily_price_file = os.path.join('data', 'daily_data.csv')

    battery_specs_dict, all_data = process_data(battery_spec_file, half_hourly_price_file, daily_price_file)

    assert args.s_init >= 0 and args.s_init <= battery_specs_dict['V_init']

    # Run simulation
    output = run_simulation(battery_specs_dict, 
                            all_data, 
                            planning_horizon=args.plan_horizon,
                            execution_horizon=args.exec_horizon,  
                            delta_t=0.5,
                            trading_freq=[1, 1, 48], 
                            S_init=args.s_init)
    
    # Save output
    output.to_csv(args.out_path)