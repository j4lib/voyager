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
    df = geopandas.GeoDataFrame.from_features(geojson)
    df.plot(ax=ax, zorder=10, **kwargs)

    # Add departure point and destination
    ax.scatter(x=departure_points[0][0], y=departure_points[0][1], color="red")
    ax.scatter(x=destination[0], y=destination[1], color="green")

    ax.set_title(f'Trip duration: {df.duration.values[0]} hours.')

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
lon_min = 5.692326 #4
lat_min = 53.671019 #52
lon_max = 13.536054 #15
lat_max = 59.388759 #60
start_date = '1995-03-01'
end_date = '1995-03-31'
follows_route = True
weights = [5, 5, 1, 100] # Interesting: [5, 5, 1, 100] # Victor's: [100, 50, 1, 100]
iterations = [15, 5, 3, 1]

# Model options
tolerance = 0.001
sigma = 0 # 100

# Trajectory options
launch_freq = 3 # days
duration = 5 # max duration in days
timestep = 900 # s
mode = 'paddling' # or 'drift', 'paddling', 'sailing'
craft = 'hjortspring' # the ones in the config
vessel_weight = 2000 # in kg
number_of_paddlers = 16
rowing_cadence = 50
oar_depth = 0 # in cm. If 0, there is no oar

destination = [8.0888, 56.7981]  # lon lat format
departure_points = [[6.4902, 58.0128]] #[[8.5693, 57.1543]]

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


#%%
# f, ax = plot_contours(chart)
# plt.show()


#%%  
# Create the model that steps throught time
model = voyager.Model(duration, timestep, sigma=sigma, tolerance=tolerance)

#%%
# Calculate the trajectories

single_result = voyager.Traverser.trajectory(mode = mode,
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

#%%
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
#%%
f, ax = plot_multiple(results, bbox, show_route=False)
plt.show()

# %%
f, ax = plot(single_result, bbox, show_route=follows_route) #, show_route=True)
plt.show()

# %%
