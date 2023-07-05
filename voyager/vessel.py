
from . import geo, search, chart
import numpy as np
import pandas as pd
from typing import *

class Vessel:
    """Class used to represent a vessel travelling in the simulation. It tracks its parameters and its travel.

    Attributes:
        x (float): last recorded position of the vessel on the x-axis (longitude)
        y (float): last recorded position of the vessel on the y-axis (latitude)
        craft (int or str): The craft type.
        mode (str): mode of navigation ('drift', 'paddling' or 'sailing'). 
        route (List[Tuple[float, float]]): ideal navigation route as calculated with A-star algorithm.
        destination (list): Destination coordinates in WGS84.
        launch_date (pd.Timestamp): Date of launch of the current vessel.
        speed (int): Paddling speed in m/s.
        paddlers (int): number of paddlers on the boat. 
        weight (int): total weight of the boat in kg. 
        cadence (int): number of strokes per minutes while paddling.
        oar_depth (int): depth of the oar, influencing variability in the trajectory, measured in cm. Depth of 0 means no oars.
        with_route (bool): Whether it uses the ideal route, or points straight to destination. 
        trajectory (List[Tuple[float, float]]): record of current trajectory.
        distance (float): distance navigated up to current time in km.
        mean_speed (float): mean_speed of navigation up to the current time in km/h.
        duration (float): duration up to current time in timesteps.
        encountered_current (List[Tuple[float, float]]): record of currents encountered up to the current time.
        encountered_winds (List[Tuple[float, float]]): record of winds encountered up to the current time.
        stops (List[str]): list of stops done by the vessels (used only when navigating multi-day).
        coord_stops (List[Tuple[float, float]]): positions of all the stops (used only when navigating multi-day).
        target (Tuple[float, float]): current target (either node in the route, or destination, depending on value of with_route).
        params (Dict): other vessel features.
        
    Methods:
        from_position(point, chart, destination, route, interval , **kwargs): Creates a vessel from a start position.
        from_positions(point, chart, destination, route, interval , **kwargs): Generates a list of vessels from multiple positions.
        update_distance(x, y): Updates position and records it to the trajectory.
        update_mean_speed(dt): Updates the total mean speed of the trajectory from the total distance travelled.
        update_encountered_environment(current, wind): Updates the currents and winds encountered in the simulation.
        update_stops(time_stop, coord_stop): Updates when a vessel stops for the night (for multi-day trips)
        has_arrived(longitude, latitude, target_tol): Calculates whether the vessel has arrived to its destination with in a certain tolerance.
        to_dict(): Saves data into dictionary that can then be converted to GeoJSON.
        to_GeoJSON(start_date, stop_date, dt): Converts vessel data into a GeoJSON representation.
    """

    def __init__(self, x, 
                       y, 
                       craft=None, 
                       mode="drift", 
                       route = None,
                       destination = None,
                       launch_date = None,
                       speed = 0,
                       paddlers = 0,
                       weight = 0,
                       cadence = 0,
                       oar_depth = 0,
                       with_route = False,
                       params = {}
                       ):
        """                
        x (float): last recorded position of the vessel on the x-axis (longitude)
        y (float): last recorded position of the vessel on the y-axis (latitude)
        craft (int or str, optional): The craft type. Defaults to 1.
        mode (str, optional): mode of navigation ('drift', 'paddling' or 'sailing'). Defaults to 'drift'.
        route (np.ndarray, optional): ideal navigation route as calculated with A-star algorithm. Defaults to None.
        destination (list, optional): Destination coordinates in WGS84. Defaults to None.
        launch_date (pd.Timestamp, optional): Date of launch of the current vessel. Defaults to None.
        speed (int, optional): Paddling speed in m/s. Defaults to 0.
        paddlers (int, optional): number of paddlers on the boat. Defaults to 0.
        weight (int, optional): total weight of the boat in kg. Defaults to 0.
        cadence (int, optional): number of strokes per minutes while paddling. Defaults to 0.
        oar_depth (int, optional): depth of the oar, influencing variability in the trajectory, measured in cm. Depth of 0 means no oars. Defaults to 0.
        with_route (bool, optional): Whether it uses the ideal route, or points straight to destination. Defaults to False.
        paramts (Dict, optional): Other features of the vessel. Defaults to {}.
        """
        
        self.craft = craft
        self.mode = mode
        self.launch_date = launch_date
        self.destination = destination
        self.x = x
        self.y = y
        self.speed = speed
        self.paddlers = paddlers
        self.weight = weight
        self.cadence = cadence
        self.oar_depth = oar_depth
        self.with_route = with_route
        
        # Initialize parameters to save
        self.trajectory = [[self.x, self.y]]
        self.distance = 0
        self.mean_speed = 0
        self.duration = 0
        self.encountered_current = []
        self.encountered_winds = []
        self.encountered_waves = []
        self.stops = []
        self.coord_stops = []

        self.route  = route

        try:
            if with_route == True:
                self.route_taken = [[float(x),float(y)] for x,y in self.route]
                self.target = self.route.pop()
            elif with_route == False:
                self.route_taken = []
                self.target = self.destination
        except:
            print("You need to specify whether a route is taken!")
            
        # Read the features of the vessel
        self.params = params


    @classmethod
    def from_position(cls, point: Tuple[float, float], 
                           chart: chart.Chart = None, 
                           destination: Tuple[float, float] = None, 
                           route: List[Tuple[float, float]] = None,
                           interval: int = 5, 
                           **kwargs):
        """Creates a vessel from a start position, using a pre-supplied Chart object and destination.

        The chart and interval parameters are used to create a route from the start position and the destination, the interval
        deciding the number of milestones along the way.

        Args:
            point (Tuple[float, float]): Start position
            chart (chart.Chart, optional): A Chart object. Defaults to None.
            destination (Tuple[float, float], optional): Destination position. Defaults to None.
            interval (int, optional): Interval to create route targets. Defaults to 5.

        Raises:
            RuntimeError: Raised if there is no possible route between start and end

        Returns:
            Vessel: A Vessel instance
        """
        
        x, y = point

        if (destination is not None) and (chart is not None):
            # if route was not supplied (is None) and with_route = True, generate a route with Astar:
            if (route is None) and kwargs['with_route']:
                # Find the closest latlon to the start and destination
                i = geo.closest_coordinate_index(chart.longitudes, x)
                j = geo.closest_coordinate_index(chart.latitudes, y)

                i_goal = geo.closest_coordinate_index(chart.longitudes, destination[0])
                j_goal = geo.closest_coordinate_index(chart.latitudes, destination[1]) 

                # Find the optimal route to the target
                astar = search.Astar(chart.grid)
                came_from, cost_so_far = astar.search(start=(j, i), goal=(j_goal, i_goal))

                # Chart the route
                try:
                    route = astar.reconstruct_path(came_from, start=(j, i), goal=(j_goal, i_goal))
                    route = [(chart.longitudes[i], chart.latitudes[j]) for j, i in route]
                    route = [route[0], *route[1:-2:interval], route[-1]]
                    route.reverse()

                except Exception as e:
                    raise RuntimeError("No possible route") from e

            # Create a vessel
            vessel = cls(x, y, 
                         route = route, 
                         destination = destination, 
                         launch_date = kwargs['launch_date'],
                         craft = kwargs['craft'], 
                         mode = kwargs['mode'],
                         speed = kwargs['speed'],
                         paddlers = kwargs['paddlers'],
                         weight = kwargs['weight'],
                         cadence = kwargs['cadence'],
                         oar_depth = kwargs['oar_depth'],
                         with_route = kwargs['with_route'])

        else:
            # Create a vessel without a route
            vessel = cls(x, y, destination=destination, **kwargs)

        return vessel


    @classmethod
    def from_positions(cls, points: List[Tuple[float, float]], 
                            chart: chart.Chart = None, 
                            destination: Tuple[float, float] = None, 
                            route: List[Tuple[float, float]] = None,
                            interval: int = 5, 
                            **kwargs) -> List:
        """Generates a list of vessels from multiple positions.
            The chart and interval parameters are used to create a route from the start position and the destination, the interval
            deciding the number of milestones along the way.

        Args:
            points (List[Tuple[float, float]]): List of positions
            chart (chart.Chart, optional): A chart object. Defaults to None.
            destination (Tuple[float, float], optional): Destination coordinates in WGS84. Defaults to None.
            interval (int, optional): Interval to create route targets. Defaults to 5.

        Returns:
            List: List of Vessel instances
        """
        vessels = []
        for point in points:

            vessel = cls.from_position(point, 
                                       chart, 
                                       destination, 
                                       route,
                                       interval, 
                                       **kwargs)

            vessels.append(vessel)


        return vessels

    def update_position(self, x: float, y: float):
        """Updates position and records it to the trajectory

        Args:
            x (float): longitudinal position
            y (float): latitudinal position
        """
            
        self.x = x
        self.y = y

        # Record position to trajectory
        self.trajectory.append([x, y])

        return self

    def update_distance(self, dx: float, dy: float):
        """Updates the cumulative distance travelled for the trajectory, as well as duration (in time steps)

        Args:
            dx (float): longitudinal displacement (km)
            dy (float): latitudinal displacement (km)
        """

        if dx == 0.0 and dy == 0.0:
            self.distance += 0.0
            self.duration += 1
            
            return self

        self.distance += np.linalg.norm(np.vstack((dx.squeeze(), dy.squeeze()))).squeeze().item()
        self.duration += 1

        return self

    def update_mean_speed(self, dt: float):
        """Updates the total mean speed of the trajectory
        from the total distance travelled.

        Args:
            dt (float): time step during the simulation (s)
        """

        N_SECONDS_PER_HOUR = 3600
        self.mean_speed = self.distance / (len(self.trajectory) * dt / N_SECONDS_PER_HOUR) # km/h

        return self

    def update_encountered_environment(self, current: Tuple[float, float], wind: Tuple[float, float], waves: float):
        """Updates the currents and winds encountered in the simulation.

        Args:
            current (List[float, float]): horizontal and vertical current at each update
            wind (List[float, float]): horizontal and vertical wind at each update
        """
        self.encountered_current.append([current[0], current[1]])
        self.encountered_winds.append([wind[0], wind[1]])
        self.encountered_waves.append(waves)


        return self
    
    def update_stops(self, stop_date: pd.Timedelta, stop_coord: Tuple[float, float]):
        """When a vessel stops for the night, it saves the time and position of the stop.

        Args:
            stop_date (pd.TimeDelta): time of stop for the night
            stop_coord (Tuple[float, float]): coordinates of stop for the night
        """
        self.stops.append(stop_date)
        self.coord_stops.append(stop_coord)

        return self

    def has_arrived(self, longitude: float, latitude: float, target_tol: float) -> bool:
        """Calculates whether the vessel has arrived to its destination with in a certain tolerance.

        Args:
            longitude (float): Longutide
            latitude (float): Latitude
            target_tol (float): Distance tolerance away from the target destination

        Returns:
            bool: Whether the vessel has arrived or not
        """

        is_close = geo.distance((longitude, latitude), self.target) <= target_tol

        if is_close:
            if self.with_route:
                if len(self.route) > 0:
                    self.target = self.route.pop()
                else:
                    return True
            else:
                return True
        else:
            return False

    def to_dict(self):
        """Saves data into dictionary that can then be converted to GeoJSON.
        """

        return {
            "trajectory": self.trajectory,
            "distance": self.distance,
            "route": self.route_taken,
            "mean_speed": self.mean_speed,
            "destination": self.destination
        }

    def to_GeoJSON(self, start_date: str, stop_date: str, dt: float) -> Dict:
        """Converts vessel data into a GeoJSON representation

        Args:
            vessel (Vessel): A Vessel object
            start_date (str): The start date of the trajectory
            stop_date (str): The end date of the trajectory
            dt (float): Timestep

        Returns:
            Dict: A dictionary compliant with GeoJSON
        """

        format_dict = {"type": "FeatureCollection",
                    "features": []
                    }

        format_dict["features"].append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": self.trajectory,
                },
                "properties": {
                    "start_date": start_date,
                    "stop_date": stop_date,
                    "timestep": dt,
                    "distance": self.distance,
                    "mean_speed": self.mean_speed,
                    "destination": self.destination,
                    "route": self.route_taken,
                    "duration": self.duration*dt / 3600, # in hours
                    "trip_currents": self.encountered_current,
                    "trip_winds": self.encountered_winds,
                    "trip_waves": self.encountered_waves,
                    "stop_times": self.stops,
                    "stop_coords": self.coord_stops,
                }          
            }
        )

        return format_dict