import voyager
import pandas as pd
import yaml
import json
import matplotlib.pyplot as plt

from plotting_tool import plot_multiple


def load_yaml(file):

    with open(file, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    return config


data_directory = "/media/mtomasini/LaCie/LIR/"
vessel_cfg_path = "./voyager/configs/vessels.yml"
replicates = 1


# Chart options
lon_min = 5.692326
lat_min = 53.671019
lon_max = 13.536054
lat_max = 59.388759
start_date = '1993-01-01'
end_date = '1994-12-31'

# Model options
tolerance = 0.001
sigma = 500
follows_route = False

# Trajectory options
launch_freq = 1 # days
duration = 3 # max duration in days
timestep = 900 # in seconds, 900 s = 15 minutes
mode = 'paddling'
craft = 'hjortspring' # the ones in the config
vessel_weight = 3000 # in kg
number_of_paddlers = 20
rowing_cadence = 70
oar_depth = 100 # in cm. If 0, there is no oar

destination = [6.6024, 58.0317]  # lon lat format
departure_points = [[8.5693, 57.1543]] # 

##### SIMULATION INITIALIZATION

# Create the bounding box, observe the order (lonlat)
bbox = [lon_min, lat_min, lon_max, lat_max]

# Convert time from datetime to timestamp
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

# Read the vessel configurations
vessel_cfg = load_yaml(vessel_cfg_path)

# Create chart
chart = voyager.Chart(bbox, start_date, end_date + pd.Timedelta(duration, unit="days")).load(data_directory)
model = voyager.Model(duration, timestep, sigma=sigma, tolerance=tolerance)

# Run simulation
for replicate in range(1, replicates + 1):

    avg_durations = pd.DataFrame(columns = ['Start day', 'Duration', 'Sunrise', 'Sunset'])

    results = voyager.Traverser.trajectories(mode = mode,
                                            craft = craft, 
                                            duration = duration,
                                            timestep = timestep, 
                                            destination = destination, 
                                            paddlers = number_of_paddlers,
                                            weight = vessel_weight,
                                            cadence = rowing_cadence,
                                            oar_depth = oar_depth,    
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
        start_day = result['features'][0]['properties']['start_date'][:-9]

        filename = start_day + f'_{replicate}'
        with open('./results/' + filename, 'w') as file:
            json.dump(result, file, indent=4)

        data_to_append = pd.DataFrame([{
             'Start day': start_day,
             'Duration': result['features'][0]['properties']['duration'],
             'Sunrise': voyager.utils.calculate_sunrise(start_day, departure_points[0]),
             'Sunset': voyager.utils.calculate_sunrise(start_day, departure_points[0])
        }])

        avg_durations = pd.concat([avg_durations, data_to_append], ignore_index=True)

    avg_durations.to_csv(f'./results/Aggregates/replicate_{replicate}.csv', sep='\t')