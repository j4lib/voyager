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

def plot(geojson: Dict, bbox: List, **kwargs):
    """Utility function to statically visualize the calculated trajectories

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

    return fig, ax

def load_yaml(file):

    with open(file, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    return config

#%%
data_directory = "D:/LIR/"
vessel_cfg_path =  './voyager/configs/vessels.yml'

# Chart options
lon_min = 5.692326 #4
lat_min = 53.671019 #52
lon_max = 13.536054 #15
lat_max = 59.388759 #60
start_date = '2018-10-27'
end_date = '2018-10-30'
weights = [1, 1, 1, 1] # [100, 50, 1, 100]
iterations = [15, 5, 3, 1]

# Model options
tolerance = 0.001
sigma = 500 # 100

# Trajectory options
launch_freq = 2 # days
duration = 2 # max duration in days
timestep = 600 # s
mode = 'paddling' # or 'drift', 'paddling', 'sailing'
craft = 2 # the ones in the config
destination = [6.6024, 58.0317]  # lon lat format
departure_points = [[8.5237, 57.1407]] # 

# Create the bounding box, observe the order (lonlat)
bbox = [lon_min, lat_min, lon_max, lat_max]

# Convert time from datetime to timestamp
start_date = pd.Timestamp(start_date)

# Read the vessel configurations
vessel_cfg = load_yaml(vessel_cfg_path)

#%%
# Create the chart
# Should possibly be pre-computed if computation is too slow
chart = voyager.Chart(bbox, start_date, start_date+pd.Timedelta(duration, unit='days'))\
                    .load(data_directory, weights=weights, iterations=iterations)

#f, ax = plot_contours(chart)
#plt.show()


#%%  
# Create the model that steps throught time
model = voyager.Model(duration, timestep, sigma=sigma, tolerance=tolerance)

#%%
# Calculate the trajectories

results = voyager.Traverser.trajectory(
                        mode = mode,
                        craft = craft, 
                        duration = duration,
                        timestep = timestep, 
                        destination = destination,  
                        bbox = bbox, 
                        departure_point = departure_points[0], 
                        vessel_params=vessel_cfg,
                        chart = chart, 
                        model = model,
                        follows_route=True
                    )

#%%
f, ax = plot(results['features'], bbox)
plt.show()
# %%
results
# %%
