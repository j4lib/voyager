import voyager
import pandas as pd
import yaml
import json


def load_yaml(file):

    with open(file, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    return config


data_directory = "D:/LIR/"
vessel_cfg_path = "./voyager/configs/vessels.yml"

# Chart options

lon_min = 5.692326
lat_min = 53.671019
lon_max = 13.536054
lat_max = 59.388759
start_date = '1993-03-03'
end_date = '1993-03-05'

# Model options
tolerance = 0.001
sigma = 1000
follows_route = False

# Trajectory options
launch_freq = 1 # days
duration = 3 # max duration in days
timestep = 900 # in seconds, 900 s = 15 minutes
mode = 'paddling'
craft = 'hjortspring' # the ones in the config
departure_points = [[8.5237, 57.1407]]
destination = [6.6024, 58.0317]  # lon lat format

##### SIMULATION INITIALIZATION

# Create the bounding box, observe the order (lonlat)
bbox = [lon_min, lat_min, lon_max, lat_max]

# Convert time from datetime to timestamp
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

# Read the vessel configurations
vessel_cfg = load_yaml(vessel_cfg_path)

# Create chart
chart = voyager.Chart(bbox, start_date, end_date).load(data_directory)
model = voyager.Model(duration, timestep, sigma=sigma, tolerance=tolerance)

results = voyager.Traverser.trajectories(mode = mode,
                                        craft = craft, 
                                        duration = duration,
                                        timestep = timestep, 
                                        destination = destination,  
                                        start_date = start_date,
                                        end_date = end_date,
                                        bbox = bbox, 
                                        departure_point = departure_points[0],
                                        vessel_params=vessel_cfg,
                                        launch_day_frequency = launch_freq,
                                        chart = chart, 
                                        model = model,
                                        follows_route = follows_route)

for result in results:
    filename = result['features'][0]['properties']['start_date']
    with open('./results/' + filename, 'w') as file:
        json.dump(result, file, indent=4)