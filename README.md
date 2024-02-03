# battery_optimization

This repository contains the code for a battery optimization model that maximizes the profit obtained by charging and discharging the battery and trading the electricity across several electricity markets

# Structure of directory
```
|- data/
    |- battery_specs.csv
    |- daily_data.csv
    |- half_hourly_data.csv
|- output/
    |- output.csv
|- src/
    |- battery_model.py
    |- data_processing.py
|- demo_notebook.ipynb
|- requirements.yml
|- run.py
``` 

* ```data/``` folder contains the csv files for battery specifications, and electricity prices from 2018-2020 of 2 half-hourly traded markets and 1 daily traded market
* ```output/``` folder contains the output of the simulation over the entire period in csv file. The file is generated as output of ```run.py```.
* ```src/``` folder contains code for data processing and the battery model class
* ```demo_notebook.ipynb``` is a jupyter notebook for interactive analysis
* ```requirements.yml``` is the conda requirement file for environment setup
* ```run.py``` is the main code to run the simulation

# Running the code

1. Clone the repository
```
git clone https://github.com/zhoutianxun/battery_optimization.git
cd battery_optimization
```

2. Set up python environment with conda
```
conda env create -f requirements.yml
conda activate energy_modeling
```

3. To run the model over the time period provided in data/, with default settings:
```
python run.py
```
To change the settings of the simulation:
* planning horizon length (time unit)
* execution horizon length (time unit)
* initial charge state of battery (MWh)
* save path for simulation output

change the following arguments:
```
usage: run.py [-h] [--plan_horizon PLAN_HORIZON] [--exec_horizon EXEC_HORIZON] [--s_init S_INIT] [--out_path OUT_PATH]

options:
  -h, --help            show this help message and exit
  --plan_horizon PLAN_HORIZON, -p PLAN_HORIZON
                        planning horizon (default=144)
  --exec_horizon EXEC_HORIZON, -e EXEC_HORIZON
                        execultion horizon (default=48)
  --s_init S_INIT, -s S_INIT
                        initial state of charge of battery (default=0)
  --out_path OUT_PATH, -o OUT_PATH
                        path for saving output (default="output/output.csv")
```
4. To run the model in interactive manner, a jupyter notebook "demo_notebook.ipynb" is provided
