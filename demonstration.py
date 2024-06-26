#%%
import voyager
import pandas as pd
import yaml
from typing import *

import cartopy
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

def plot_contours(chart: voyager.Chart):
    data = chart.grid.weighted_mask
    data[np.isnan(data)] = 1000
    # data = chart.grid.weighted_mask

    values = np.unique(data)
    cmap = plt.get_cmap('PuBu_r', np.max(data) - np.min(data) + 1)

    fig, ax = plt.subplots(figsize=(20,10))
    im = ax.imshow(data, cmap=cmap,  norm=colors.LogNorm(vmin=values.min()+1, vmax=values.max()))
    ax.invert_yaxis()
    fig.colorbar(im, ax=ax, extend='max')

    ax.set_title("Chart with shoreline contours")

    return fig, ax

def plot(geojson: Dict, bbox: List, show_route: bool = False, **kwargs):
    """Utility function to statically visualize one of the calculated trajectories

        Args:
            trajectory_file (str, optional): The GeoJSON files with trajectories. Defaults to None.

        Returns:
            fig, ax: Matplotlib figure and axis tuples
        """
        
    # Create matplotlib figure objects
    fig, ax = plt.subplots(subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(20,10))

    # Choose coastline resolution
    ax.coastlines('50m')

    # Limit map to bounding box
    ax.set_extent([bbox[0], bbox[2], bbox[1], bbox[3]], cartopy.crs.PlateCarree())

    # Add ocean and land features, for visuals
    ax.add_feature(cartopy.feature.OCEAN, zorder=0)
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')

    # Adds gridds to visual
    ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.2, linestyle='--')

    # Use geopandas built-in GeoJSON processing and visualization
    df = geopandas.GeoDataFrame.from_features(geojson["features"])
    df.plot(ax=ax, zorder=10, **kwargs)

    # Add departure point and destination
    ax.scatter(x=departure_points[0][0], y=departure_points[0][1], color="red")
    ax.scatter(x=destination[0], y=destination[1], color="green")
    ax.scatter(x=11.2, y=58.6, color="purple")

    stop_coordinates = pd.DataFrame(geojson['features'][0]['properties']['stop_coords'], columns=['x', 'y'])
    ax.scatter(x=stop_coordinates['x'], y=stop_coordinates['y'], color="gold",s = 100)

    ax.set_title(f'Effective duration: {df.duration.values[0]}h')

    if show_route == True:
        for i in range(len(df.route[0])):
            ax.scatter(x=df.route[0][i][0], y=df.route[0][i][1], color="blue")

    return fig, ax


def plot_multiple(geojson_list: List[Dict], bbox: List, show_route: bool = False, **kwargs):
    """Utility function to statically visualize all the calculated trajectories

    Args:
        geojson: List[Dict]
        bbox: List
        show_route: bool

    Returns:
        fig, ax: Matplotlib figure and axis tuples
    """
        
    # Create matplotlib figure objects
    fig, ax = plt.subplots(subplot_kw={'projection': cartopy.crs.PlateCarree()}, figsize=(20,10))

    # Choose coastline resolution
    ax.coastlines('50m')

    # Limit map to bounding box
    ax.set_extent([bbox[0], bbox[2], bbox[1], bbox[3]], cartopy.crs.PlateCarree())

    # Add ocean and land features, for visuals
    ax.add_feature(cartopy.feature.OCEAN, zorder=0)
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')

    # Adds gridds to visual
    ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.2, linestyle='--')

    # calculate average duration:
    durations = []
    # Use geopandas built-in GeoJSON processing and visualization
    for geojson in geojson_list:
        df = geopandas.GeoDataFrame.from_features(geojson['features'])
        df.plot(ax=ax, zorder=10, **kwargs)
        durations.append(df.duration.values[0])

    # Add departure point and destination
    ax.scatter(x=departure_points[0][0], y=departure_points[0][1], color="red")
    ax.scatter(x=destination[0], y=destination[1], color="green")

    ax.set_title(f'Mean trip duration: {round(np.mean(durations), 2)} hours.')

    if show_route == True:
        df = geopandas.GeoDataFrame.from_features(geojson_list[0])
        for i in range(len(df.route[0])):
            ax.scatter(x=df.route[0][i][0], y=df.route[0][i][1], color="blue")

    return fig, ax


def load_yaml(file):

    with open(file, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    return config

#%%
data_directory = "/media/mtomasini/LaCie/LIR/"
vessel_cfg_path =  './voyager/configs/vessels.yml'

# Chart options
lon_min = -7.343910
lat_min = 49.585907
lon_max = 9.356939
lat_max = 57.402805
start_date = '1993-03-23'
end_date = '1993-03-25'
follows_route = True

# weights indicate the actual weight of each weight layer (from out to sea to coast)
# iterations indicate how large the layers are
weights = [1, 10, 1, 100] # [5, 5, 1, 100] # Interesting: [5, 5, 1, 100] # Neutral: [1,1,1,1]
iterations = [10, 5, 3, 1] # alternative [10, 8, 4, 2]

# Model options
tolerance = 0.001
location_sigma = 0 # 100
angle_sigma = 0

# Trajectory options
launch_freq = 10 # days
duration = 30 # max duration in days
timestep = 3600 # 900 s
mode = 'paddling' # or 'drift', 'paddling', 'sailing'
craft = 'hjortspring' # the ones in the config
vessel_weight = 3000 # in kg
number_of_paddlers = 16
rowing_cadence = 44
oar_depth = 75 # in cm. If 0, there is no oar


destination = [-3.82, 53.32] # [7.3663, 57.9517] # lon lat format
departure_points = [[1.36, 51.32]] # 

# route = [destination,
#          [11.3489, 57.8185],
#          [11.200, 57.9661],
#          [10.9863, 58.4506],
#          [10.8875, 58.7518], 
#          [10.573, 58.9017],
#          [10.3271, 58.8336], #
#          [10.3049, 58.8149],
#          [9.9106, 58.6795],
#          [9.5103, 58.5235],
#          [9.2545, 58.3084],
#          [8.9420, 58.0970],
#          [8.5955, 58.0201],
#          [8.4098, 57.8600],
#          [8.1128, 57.7719],
#          [7.7199, 57.7694],
#          departure_points[0]]

# Create the bounding box, observe the order (lonlat)
bbox = [lon_min, lat_min, lon_max, lat_max]

# Convert time from datetime to timestamp
start_date = pd.Timestamp(start_date)
#start_date = voyager.utils.calculate_sunrise(start_date, departure_points[0])
end_date = pd.Timestamp(end_date)

# Read the vessel configurations
vessel_cfg = load_yaml(vessel_cfg_path)

#%%
# Create the chart
# Should possibly be pre-computed if computation is too slow
chart = voyager.Chart(bbox, start_date, end_date + pd.Timedelta(duration, unit="days"))\
                    .load(data_directory, weights=weights, iterations=iterations)

departure_points = [chart.find_closest_water(departure_points[0][0], departure_points[0][1], radar_radius=1)]
destination = chart.find_closest_water(destination[0], destination[1], radar_radius=1)
route = [[-3.777940034866333, 53.33399963378906], 
         [-4.111269950866699, 53.46733856201172], 
         [-4.338838, 53.515715], 
         [-4.766820049285889, 53.46733856201172],
         [-4.666820049285889, 53.13399124145508], 
         [-4.889039993286133, 52.80064010620117],
         [-4.333489894866943, 52.533958435058594],
         [-4.889039993286133, 52.26728057861328],
         [-5.333479881286621, 51.933929443359375], 
         [-5.222370147705078, 51.60057830810547], 
         [-4.666820049285889, 51.46723937988281], 
         [-4.5557098388671875, 51.133888244628906], 
         [-4.889039993286133, 50.800540924072266], 
         [-5.444590091705322, 50.46718978881836], 
         [-6.013905, 50.13383865356445],
         [-5.555699825286865, 49.93383026123047], 
         [-5.000150203704834, 49.93383026123047], 
         [-4.4446001052856445, 50.20050811767578],
         [-3.889050006866455, 50.13383865356445], 
         [-3.3334999084472656, 50.267181396484375], 
         [-2.7779500484466553, 50.600528717041016], 
         [-2.222399950027466, 50.46718978881836], 
         [-1.666849970817566, 50.50718978881836],
         [-1.111299991607666, 50.46718978881836], 
         [-0.5557500123977661, 50.667198181152344],
         [-0.00019999999494757503, 50.667198181152344],
         [0.5553500056266785, 50.73387145996094],
         [1.36, 50.9],
         [1.9, 51.3]]
# route = list(reversed(route))

#%%
# f, ax = plot_contours(chart)
# plt.show(block=False)

#%%  
# Create the model that steps throught time
model = voyager.Model(duration, timestep, sigma=location_sigma, angle_sigma=angle_sigma, tolerance=tolerance)

#%%
# Calculate the trajectories

# IMPORTANT! use route.copy() to avoid route getting cancelled by an internal pop() method.
single_result = voyager.Traverser.trajectory_by_day(mode = mode,
                                             craft = craft, 
                                             duration = duration,
                                             timestep = timestep, 
                                             destination = destination,
                                             start_date = start_date,
                                             paddlers = number_of_paddlers,
                                             weight = vessel_weight,
                                             cadence = rowing_cadence,
                                             oar_depth = oar_depth,
                                             bbox = bbox,
                                             route = route.copy(),
                                             departure_point = departure_points[0], 
                                             vessel_params=vessel_cfg,
                                             chart = chart, 
                                             model = model,
                                             follows_route = follows_route)

#%%
# import json 
# with open('./test.json', 'w') as file:
#     json.dump(single_result, file, indent=4)

#%%
f, ax = plot(single_result, bbox, show_route=follows_route)
plt.show()

#%%
# results = voyager.Traverser.trajectories(mode = mode,
#                                         craft = craft, 
#                                         duration = duration,
#                                         timestep = timestep, 
#                                         destination = destination,  
#                                         paddlers = number_of_paddlers,
#                                         weight = vessel_weight,
#                                         cadence = rowing_cadence,
#                                         oar_depth = oar_depth,   
#                                         start_date = start_date,
#                                         end_date = end_date,
#                                         bbox = bbox, 
#                                         departure_point = departure_points[0],
#                                         vessel_params=vessel_cfg,
#                                         launch_day_frequency = launch_freq,
#                                         chart = chart, 
#                                         model = model,
#                                         follows_route = follows_route)
#%%
# f, ax = plot_multiple(results, bbox, show_route=follows_route)
# plt.show()

# %%
