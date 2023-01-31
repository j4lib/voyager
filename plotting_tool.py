#%%
import voyager
import pandas as pd
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

def plot(geojson: Dict, bbox: List, departure_points: List, destination: List, show_route: bool = False, **kwargs):
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


def plot_multiple(geojson_list: List[Dict], bbox: List, departure_points: List, destination: List, show_route: bool = False, **kwargs):
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
#%% 
