import pandas as pd

def process_data(battery_spec_file, 
                 half_hourly_price_file, 
                 daily_price_file) -> (dict, pd.DataFrame):
    
    """
    process all input data files into output variables for input into battery model

    Parameters
    ----------
    battery_spec_file : str
        path to csv file containing battery specification
    half_hourly_price_file : str
        path to csv file containing half hourly electricity market prices
    daily_price_file : str
        path to csv file containing daily electricity market prices    

    Returns
    -------
    battery_specs_dict : dict
        dictionary of battery specs 
    all_data : pd.DataFrame
        combined price file for both half hourly and daily electricity market prices
    """

    # Read battery_spec_file
    battery_specs = pd.read_csv(battery_spec_file, index_col=0)
    battery_specs_dict = {}
    battery_specs_dict['K_C'] = float(battery_specs.loc['Max charging rate', 'Values']) 
    battery_specs_dict['K_D'] = float(battery_specs.loc['Max discharging rate', 'Values'])
    battery_specs_dict['V_init'] = float(battery_specs.loc['Max storage volume', 'Values'])
    battery_specs_dict['eta_C'] = 1 - float(battery_specs.loc['Battery charging efficiency', 'Values']) 
    battery_specs_dict['eta_D'] = 1 - float(battery_specs.loc['Battery discharging efficiency', 'Values'])
    battery_specs_dict['lifetime'] = float(battery_specs.loc['Lifetime (1)', 'Values'])
    battery_specs_dict['n_max'] = float(battery_specs.loc['Lifetime (2)', 'Values'])
    battery_specs_dict['f'] = float(battery_specs.loc['Storage volume degradation rate', 'Values'])/100
    battery_specs_dict['Capex'] = float(battery_specs.loc['Capex', 'Values'])

    # Read half_hourly_price_file
    half_hourly_data = pd.read_csv(half_hourly_price_file)
    half_hourly_data.loc[:, 'Datetime'] = pd.to_datetime(half_hourly_data['Datetime'], format='%m/%d/%Y %H:%M')
    half_hourly_data = half_hourly_data.iloc[:, :3]

    # Read daily_price_file
    daily_data = pd.read_csv(daily_price_file)
    daily_data.loc[:, 'Datetime'] = pd.to_datetime(daily_data['Datetime'], format='%m/%d/%Y')
    daily_data.head()

    # combine data
    all_data = half_hourly_data.merge(daily_data, on='Datetime', how='left')
    all_data['Market 3 Price [£/MWh]'] = all_data['Market 3 Price [£/MWh]'].ffill()
    all_data.head()

    return battery_specs_dict, all_data