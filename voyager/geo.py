"""Geographic tools

This script contains tools to calculate different geographic quantities in the simulation. 
It requires that the packages geopy.distance and numpy be installed within the Python environment.

This file can be imported as module and contains the following functions:
    * closest_coordinate_index - returns the index of the array closest to the given coordinate
    * lonlat_from_displacement - calculates new coordinates after displacement, given an origin
    * geodesic - algorithm to calculate displacement using a geodesic approach
    * great_circle - algorithm to calculate displacement using a great circle approach
    * distance - calculates distance in km between two lon/lat points
    * distance_from_displacement - gives distance along a displacement
    * bearing_from_displacement - gives direction of a given displacement (in degrees with respect to North) 
    * bearing_from_lonlat - gives angle between a coordinate and a target (in degrees with respect to North) 
"""


import geopy.distance as gp
import numpy as np
import math
from typing import *


def closest_coordinate_index(array, value):
    """Function to find the index of the closes coordinate.

    Args:
        array (array): array of either latitudes or longitudes (see Chart)
        value (float): single coordinate (longitude or latitude)
    Returns:
        idx (int): index of the array closest to value 
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def lonlat_from_displacement(dx: float, dy: float, origin: Tuple[float, float], method='geodesic') -> Tuple[float, float]:
    """Calculates new coordinates after displacement, given an origin.

    Args:
        dx (float): displacement in km along the x axis
        dy (float): displacement in km along the y axis
        origin (Tuple[float, float]): starting point of displacement as [lon, lat]
        method (str): either 'geodesic' or 'great_circle', determines algorithm to use to calculate coordinates
    
    Raises:
        ValueError: raised if method isn't either 'geodesic' or 'great_circle' 
    
    Returns:
        Tuple[float, float]: longitude and latitude of point reached after displacement
    
    """

    if method == 'geodesic': 
        lon, lat = geodesic(dx, dy, origin)

        return lon, lat

    elif method == 'great circle':
        lon, lat = great_circle(dx, dy, origin)

        return lon, lat

    else: 
        raise ValueError("Method must be geodesic or great circle")

def geodesic(dx: float, dy: float, origin: Tuple[float, float]) -> Tuple[float, float]:
    """Algorithm to calculate displacement using a geodesic approach.

    Args: 
        dx (float): displacement in km along the x axis
        dy (float): displacement in km along the y axis
        origin (Tuple[float, float]): starting point of displacement as [lon, lat]

    Returns:
        Tuple[float, float]: longitude and latitude of point reached after displacement with geodesic.
    """

    # Calculate the bearing of the displacement
    bearing = bearing_from_displacement(dx, dy)

    # Calculate the distance from the displacement
    distance = distance_from_displacement(dx, dy)

    # Transforms the point to a geopy point
    start = gp.lonlat(*origin)

    # Using WGS-84 per default
    destination = (gp.distance(kilometers=distance)
                      .destination(point=start, bearing=bearing))

    return destination.longitude, destination.latitude


def great_circle(dx: float, dy: float, origin: Tuple[float, float]) -> Tuple[float, float]:
    """Algorithm to calculate displacement using a great circle approach.

    Args: 
        dx (float): displacement in km along the x axis
        dy (float): displacement in km along the y axis
        origin (Tuple[float, float]): starting point of displacement as [lon, lat]

    Returns:
        Tuple[float, float]: longitude and latitude of point reached after displacement with great circle.
    """

    longitude, latitude = origin

    r_earth = 6371 # km

    new_latitude  = latitude  + (dy / r_earth) * (180 / np.pi)
    new_longitude = longitude + (dx / r_earth) * (180 / np.pi) / np.cos(latitude * np.pi/180)

    return new_longitude.item(), new_latitude.item()


def distance(origin: Tuple[float, float], target: Tuple[float, float]) -> float:
    """Calculates distance in km between two lon/lat points, in km

    Args:
        origin (Tuple[float, float]): point of origin as [lon, lat]
        target (Tuple[float, float]): point of destination as [lon, lat]

    Returns:
        float: distance between origin and target
    """
    return gp.distance(gp.lonlat(*origin), gp.lonlat(*target)).km

def distance_from_displacement(dx: float, dy: float) -> float:
    """Gives distance along a displacement

    Args:
        dx (float): displacement in km along the x axis
        dy (float): displacement in km along the y axis
    Returns:
        float: distance over a displacement
    """

    return np.linalg.norm(np.array((dx, dy)))

def bearing_from_displacement(dx: float, dy: float) -> float:
    """Gives direction of a given displacement (in degrees with respect to North) 
    
    Args:
        dx (float): displacement in km along the x axis
        dy (float): displacement in km along the y axis
    Returns:
        float: angle (in degrees) of a displacement on the map 
    """

    angle = np.rad2deg(np.arctan2(dy, dx))

    bearing = (90 - angle)

    return bearing

def bearing_from_lonlat(position: np.ndarray, target: np.ndarray) -> float:
    """Gives angle between a coordinate and a target (in degrees with respect to North) 

    Args:
        dx (float): displacement in km along the x axis
        dy (float): displacement in km along the y axis
    Returns:
        float: bearing (angle) between a position and a target in degrees
    """

    lat_pos = np.deg2rad(position[1])
    lat_tgt = np.deg2rad(target[1])
    d_lon = np.deg2rad(position[0]-target[0])

    x = np.sin(d_lon) * np.cos(lat_tgt)
    y = np.cos(lat_pos) * np.sin(lat_tgt) - np.sin(lat_pos) * np.cos(lat_tgt) * np.cos(d_lon)

    bearing = np.arctan2(x, y)

    bearing = (np.rad2deg(bearing) + 360) % 360

    return bearing