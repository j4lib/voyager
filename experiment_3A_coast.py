"""
EXPERIMENT 3A

Author: Matteo Tomasini
Date: 04.04.2023

In experiment 3A we simulate a trip starting from West Thy, crossing the Skagerrak, then navigating along the the West Coast of Sweden to Lista.
In this part of the script we simulate the coastal part of the trip (from the West Coast of Sweden to Lista).

The chosen boat is the Hjortspring boat, with 16 paddlers paddling at 50 strokes per minute, weighing 3000 kg, and 
with steering oars 75 cm deep.
"""

import os
import voyager
import pandas as pd
import yaml
import json
import matplotlib.pyplot as plt
import time

from plotting_tool import plot_multiple

def load_yaml(file):

    with open(file, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    return config


high_wave_limit = 2 # what height of waves do we consider "stormy"? (in meters)

data_directory = "/media/mtomasini/LaCie/LIR/"
vessel_cfg_path = "./voyager/configs/vessels.yml"
replicates = 1

# Chart options
lon_min = 5.692326
lat_min = 53.671019
lon_max = 13.536054
lat_max = 59.388759
start_date = '1993-01-02' # ! If 1993, trips start on Jan 2, since Jan 1 has data starting only at noon
end_date = '1993-01-31'

# Model options
tolerance = 0.001
sigma = 0
follows_route = False

# Trajectory options
launch_freq = 1 # days
duration = 5 # max duration in days
timestep = 900 # in seconds, 900 s = 15 minutes
mode = 'paddling'
craft = 'hjortspring' # the ones in the config
vessel_weight = 3000 # in kg
number_of_paddlers = 16
rowing_cadence = 50
oar_depth = 75 # in cm. If 0, there is no oar

destination = [6.6024, 58.0317]  # lon lat format
departure_points = [[8.5693, 57.1543]] # 

##### SIMULATION INITIALIZATION

# Create the bounding box, observe the order (lonlat)
bbox = [lon_min, lat_min, lon_max, lat_max]

# Convert time from datetime to timestamp
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date) # end date is the last date we will simulate

simulated_dates = pd.date_range(start_date, end_date, freq = f'{launch_freq}D')

# Read the vessel configurations
vessel_cfg = load_yaml(vessel_cfg_path)

# Run simulation
start_time = time.time()
for replicate in range(1, replicates + 1):
    # create empty dataframe for aggregates
    avg_durations = pd.DataFrame(columns = ['Start day', 'Duration', 'Sunrise', 'Sunset'])

    for date in simulated_dates:
        stop_date = date + pd.Timedelta(duration, unit="days") # date at which the replicate stops

        # Create chart
        chart = voyager.Chart(bbox, date, stop_date).load(data_directory)
        model = voyager.Model(duration, timestep, sigma=sigma, tolerance=tolerance)

        trajectory = voyager.Traverser.trajectory(mode = mode,
                                                craft = craft, 
                                                duration = duration,
                                                timestep = timestep, 
                                                destination = destination, 
                                                paddlers = number_of_paddlers,
                                                weight = vessel_weight,
                                                cadence = rowing_cadence,
                                                oar_depth = oar_depth,    
                                                bbox = bbox, 
                                                departure_point = departure_points[0],
                                                vessel_params=vessel_cfg,
                                                chart = chart, 
                                                model = model,
                                                follows_route = follows_route)
        
        filename = date.strftime('%Y-%m-%d') + f'_{replicate}'
        with open(data_directory + '/results/Experiment1A/' + filename, 'w') as file:
            json.dump(trajectory, file, indent=4)

        trajectory_data = trajectory['features'][0]['properties']

        # calculate for how many hours high waves were encountered:
        high_waves_count = sum(1 for i in trajectory_data['trip_waves'] if (i > high_wave_limit))
        duration_in_high_waves = high_waves_count*timestep/3600     # time in hours 

        data_to_append = pd.DataFrame([{
             'Start day': date,
             'Duration': trajectory_data['duration'],
             'HeighestWave': max(trajectory_data['trip_waves']),
             'HoursOfStorm': duration_in_high_waves,
             'Sunrise': voyager.utils.calculate_sunrise(date, departure_points[0]),
             'Sunset': voyager.utils.calculate_sunset(date, departure_points[0])
        }])

        if os.path.exists(data_directory + f'/results/Experiment1A/Aggregates/replicate_{replicate}.csv'):
            data_to_append.to_csv(data_directory + f'/results/Experiment1A/Aggregates/replicate_{replicate}.csv', mode='a', sep='\t', header=False, index=False)
        else:
            data_to_append.to_csv(data_directory + f'/results/Experiment1A/Aggregates/replicate_{replicate}.csv', mode='w', sep='\t', header=True, index=False)

    
end_time = time.time()
total_time = end_time - start_time
total_time = time.strftime("%H:%M:%S", time.gmtime(total_time))

print("It took " + total_time + " to perform this simulation.")
