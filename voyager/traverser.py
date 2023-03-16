import multiprocessing as mp

import pandas as pd
from .chart import Chart
from .models import Vessel, Model
from . import utils
from typing import *

class Traverser:
    """A class to represent the trajectory of a vessel with the chosen Chart and Model.

    Attributes:
        mode (str): mode of movement ('drift', 'paddling' or 'sailing')
        craft (str or int): type of craft (e.g. 'hjortspring')
        duration (int): maximal duration of simulation
        dt (float): time between steps in simulation
        vessel_config (str): path to vessel configuration file
        follows_route (bool): whether it follows an ideal route calculated with A-star algorithm
        destination (Tuple[float, float]): destination point as [lon, lat]
        speed (float): user supplied paddle speed (overridden in certain cases)
        data_directory (str): path to data directory
        start_date (pd.Timestamp): first day of simulation
        end_date (pd.Timestamp): last day of simulation
        dates (pd.date_range): sequence of all dates between start date and end date
        launch_day_freq (int): number of days between launches
        bbox (List): chart box with minimal/maximal lat/lon (as [lon min, lat min, lon max, lat max])
        departure_points (List): list of couple of floats giving different departure points
    Methods:
        trajectory(...): class generating method that generates a vessel and its trajectories from a certain date.
        trajectories(...): class generating method that generates a vessel and its trajectories between two dates with a certain launch frequency.
        trajectory_by_day(...): class generating method that does the same as trajectory but stops at night.
        run(chart, model, model_kwargs, chart_kwargs): Generates a set of trajectories in a date range, with a certain launch day frequency for the vessels.
        run_mp(model_kwargs, chart_kwargs): Pseudo-parallel generation of a set of trajectories in a date range, with a certain launch day frequency for the vessels.
    """

    def __init__(self, mode = 'drift', 
                       craft = 1, 
                       duration = 60,
                       timestep = 1, 
                       destination = [], 
                       speed = 2, 
                       start_date = '', 
                       end_date = '', 
                       launch_freq = 5, 
                       bbox = [], 
                       departure_points = [],
                       follows_route = False, 
                       data_directory = '', 
                       vessel_config='configs/vessels.yml') -> None:
        """
        Args:
            mode (str, optional): mode of movement ('drift', 'paddling' or 'sailing'), defaults to 'drift'.
            craft (str or int, optional): type of craft (e.g. 'hjortspring'), defaults to 1
            duration (int, optional): maximal duration of simulation, defaults to 60
            dt (float, optional): time between steps in simulation, defaults to 1
            vessel_config (str, optional): path to vessel configuration file, defaults to configs/vessels.yml
            follows_route (bool, optional): whether it follows an ideal route calculated with A-star algorithm, defaults to False
            destination (Tuple[float, float], optional): destination point as [lon, lat], defaults to []
            speed (float, optional): user supplied paddle speed (overridden in certain cases), defaults to 2
            data_directory (str, optional): path to data directory, defaults to ''
            start_date (pd.Timestamp, optional): first day of simulation, defaults to ''
            end_date (pd.Timestamp, optional): last day of simulation, defaults to ''
            launch_freq (int, optional): number of days between launches, defaults to 5
            bbox (List, optional): chart box with minimal/maximal lat/lon (as [lon min, lat min, lon max, lat max]), defaults to []
            departure_points (List, optional): list of couple of floats giving different departure points, defaults to []
        """

        self.craft      = craft
        self.mode       = mode
        self.duration   = duration
        self.dt         = timestep
        self.vessel_config = vessel_config
        self.follows_route = follows_route

        self.destination = destination
        self.speed       = speed

        self.data_directory = data_directory

        # Define datetime objects to limit simulation
        self.start_date     = pd.to_datetime(start_date, infer_datetime_format=True)
        self.end_date       = pd.to_datetime(end_date, infer_datetime_format=True) 
        self.dates          = pd.date_range(self.start_date, self.end_date) 
        
        # Interval in days to launch vessels
        self.launch_day_frequency = launch_freq
        
        # The bounding box limits the region of simulation
        self.bbox = bbox

        # Starting points for trajectories
        self.departure_points = departure_points

    @classmethod
    def trajectory(
            cls,
            mode = 'drift', 
            craft = 1, 
            duration = 60,
            timestep = 1, 
            destination = [], 
            speed = 2, 
            paddlers = 0,
            weight = 0,
            cadence = 0,
            oar_depth = 0,     
            date = '', 
            bbox = [], 
            departure_point = [], 
            data_directory = '', 
            vessel_params= {},
            chart_kwargs = {}, 
            model_kwargs = {}, 
            chart = None, 
            model = None,
            follows_route = False) -> Dict:
        """Generates a single set of trajectories from a single set of departure and destination points.

        Args:
            mode (str, optional): The mode of propulsion, either 'sailing', 'paddling' or 'drift'. Defaults to 'drift'.
            craft (int or str, optional): The craft type. Defaults to 1.
            duration (int, optional): The maximal duration in days of the trajectories. Defaults to 60.
            timestep (int, optional): Timestep for updating the speed and position of the vessels. Defaults to 1.
            destination (list, optional): Destination coordinates in WGS84. Defaults to [].
            speed (int, optional): Paddling speed in m/s. Defaults to 2.
            date (pd.Timestamp, optional): Date as a YYYY-MM-DD string. Defaults to ''.
            bbox (list, optional): Bounding box of the map. Defaults to [].
            departure_point (list, optional): Departure point in WGS84. Defaults to [].
            data_directory (str, optional): The root directory of the velocity data. Defaults to ''.
            vessel_params (dict, optional): Parameters for the vessel configuration. Defaults to {}.
            chart_kwargs (dict, optional): Parameters for the chart configuration. Defaults to {}.
            model_kwargs (dict, optional): Parameters for the model configuration. Defaults to {}.
            chart (_type_, optional): Pre-supplied Chart object. Defaults to None.
            model (_type_, optional): Pre-supplied Model object. Defaults to None.
            follows_route (bool, optional): Uses ideal route, or points to destination. Defaults to False.

        Returns:
            Dict: The trajectories as GeoJSON compliant dictionary
        """

        # The chart object keeps track of the region of interest
        # and the wind/current data for that region
        # It is shared by all vessels
        if not chart:
            start_date = pd.to_datetime(date, infer_datetime_format=True)
            end_date   = start_date + pd.Timedelta(duration, unit='days')
            chart = Chart(bbox, start_date, end_date).load(data_directory, **chart_kwargs)
        
        # The model object describes the equations of movement and
        # traversal across the oceans over time
        if not model:
            model = Model(duration, timestep, **model_kwargs)

        
        vessel = Vessel.from_position(departure_point, 
                                      craft = craft,
                                      chart = chart,
                                      destination = destination,
                                      launch_date = date,
                                      speed = speed,
                                      paddlers = paddlers,
                                      weight = weight,
                                      cadence = cadence,
                                      oar_depth = oar_depth,
                                      mode = mode,
                                      with_route = follows_route,
                                      params = vessel_params[mode][craft])

        # Interpolate the data for only the duration specified
        start_date = utils.calculate_sunrise(chart.start_date, departure_point)
        chart.interpolate(start_date, duration)

        # Use the interpolated values in the model
        model.use(chart)

        # Run the model
        vessel = model.run(vessel)

        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%S')
        stop_date_str  = (start_date + pd.Timedelta(len(vessel.trajectory)*timestep, unit='s')).strftime('%Y-%m-%dT%H:%M:%S')

        return vessel.to_GeoJSON(start_date_str, stop_date_str, timestep)

    @classmethod
    def trajectory_by_day(
            cls,
            mode = 'drift', 
            craft = 1, 
            duration = 60,
            timestep = 1, 
            destination = [], 
            speed = 2, 
            paddlers = 0,
            weight = 0,
            cadence = 0,
            oar_depth = 0,     
            date = '', 
            bbox = [], 
            departure_point = [], 
            data_directory = '', 
            vessel_params= {},
            chart_kwargs = {}, 
            model_kwargs = {}, 
            chart = None, 
            model = None,
            follows_route = False) -> Dict:
        """Generates a single set of trajectories from a single set of departure and destination 
        points, by navigating only by day.

        Args:
            mode (str, optional): The mode of propulsion, either 'sailing', 'paddling' or 'drift'. Defaults to 'drift'.
            craft (int or str, optional): The craft type. Defaults to 1.
            duration (int, optional): The maximal duration in days of the trajectories. Defaults to 60.
            timestep (int, optional): Timestep for updating the speed and position of the vessels. Defaults to 1.
            destination (list, optional): Destination coordinates in WGS84. Defaults to [].
            speed (int, optional): Paddling speed in m/s. Defaults to 2.
            date (pd.Timedelta, optional): Date as a YYYY-MM-DD string. Defaults to ''.
            bbox (list, optional): Bounding box of the map. Defaults to [].
            departure_point (list, optional): Departure point in WGS84. Defaults to [].
            data_directory (str, optional): The root directory of the velocity data. Defaults to ''.
            vessel_params (dict, optional): Parameters for the vessel configuration. Defaults to {}.
            chart_kwargs (dict, optional): Parameters for the chart configuration. Defaults to {}.
            model_kwargs (dict, optional): Parameters for the model configuration. Defaults to {}.
            chart (_type_, optional): Pre-supplied Chart object. Defaults to None.
            model (_type_, optional): Pre-supplied Model object. Defaults to None.
            follows_route (bool, optional): Uses ideal route, or points to destination. Defaults to False.

        Returns:
            Dict: The trajectories as GeoJSON compliant dictionary
        """

        # The chart object keeps track of the region of interest
        # and the wind/current data for that region
        # It is shared by all vessels
        if not chart:
            start_date = pd.to_datetime(date, infer_datetime_format=True)
            end_date   = start_date + pd.Timedelta(duration, unit='days')
            chart = Chart(bbox, start_date, end_date).load(data_directory, **chart_kwargs)
        
        # The model object describes the equations of movement and
        # traversal across the oceans over time
        if not model:
            model = Model(duration, timestep, **model_kwargs)

        vessel = Vessel.from_position(departure_point, 
                                      craft = craft,
                                      chart = chart,
                                      destination = destination,
                                      launch_date = date,
                                      speed = speed,
                                      paddlers = paddlers,
                                      weight = weight,
                                      cadence = cadence,
                                      oar_depth = oar_depth,
                                      mode = mode,
                                      with_route = follows_route,
                                      params = vessel_params[mode][craft])

        # Interpolate the data for only the duration specified
        start_date = utils.calculate_sunrise(chart.start_date, departure_point)
        chart.interpolate(start_date, duration)

        # Use the interpolated values in the model
        model.use(chart)

        # Run the model
        vessel = model.run_by_day(vessel)

        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%S')
        stop_date_str  = (start_date + pd.Timedelta(len(vessel.trajectory)*timestep, unit='s')).strftime('%Y-%m-%dT%H:%M:%S')

        return vessel.to_GeoJSON(start_date_str, stop_date_str, timestep)


    @classmethod
    def trajectories(
            cls,
            mode = 'drift', 
            craft = 1, 
            duration = 60,
            timestep = 1, 
            destination = [], 
            speed = 2, 
            paddlers = 0,
            weight = 0,
            cadence = 0,
            oar_depth = 0,
            start_date = '',
            end_date = '', 
            bbox = [], 
            departure_point = [], 
            data_directory = '', 
            vessel_params= {},
            launch_day_frequency = 5,
            chart_kwargs = {}, 
            model_kwargs = {}, 
            chart = None, 
            model = None,
            follows_route = False) -> Dict:
        """Generates trajectories from a single departure point and a destination point, with a certain launch frequency between two given dates.

        Args:
            mode (str, optional): The mode of propulsion, either 'sailing', 'paddling' or 'drift'. Defaults to 'drift'.
            craft (int, optional): The craft type. Defaults to 1.
            duration (int, optional): The maximal duration in days of the trajectories. Defaults to 60.
            timestep (int, optional): Timestep for updating the speed and position of the vessels. Defaults to 1.
            destination (list, optional): Destination coordinates in WGS84. Defaults to [].
            speed (int, optional): Paddling speed in m/s. Defaults to 2.
            date (str, optional): Date as a YYYY-MM-DD string. Defaults to ''.
            bbox (list, optional): Bounding box of the map. Defaults to [].
            departure_point (list, optional): Departure point in WGS84. Defaults to [].
            data_directory (str, optional): The root directory of the velocity data. Defaults to ''.
            vessel_params (dict, optional): Parameters for the vessel configuration. Defaults to {}.
            chart_kwargs (dict, optional): Parameters for the chart configuration. Defaults to {}.
            model_kwargs (dict, optional): Parameters for the model configuration. Defaults to {}.
            chart (_type_, optional): Pre-supplied Chart object. Defaults to None.
            model (_type_, optional): Pre-supplied Model object. Defaults to None.
            follows_route (bool, optional): Uses ideal route, or points to destination. Defaults to False.

        Returns:
            Dict: The trajectories as GeoJSON compliant dictionary
        """

        # The chart object keeps track of the region of interest
        # and the wind/current data for that region
        # It is shared by all vessels
        if not chart:
            start_date = pd.to_datetime(start_date, infer_datetime_format=True)
            max_end_date   = start_date + pd.Timedelta(duration, unit='days')
            chart = Chart(bbox, start_date, max_end_date).load(data_directory, **chart_kwargs)
        
        # The model object describes the equations of movement and
        # traversal across the oceans over time
        if not model:
            model = Model(duration, timestep, **model_kwargs)

        start_date     = pd.to_datetime(start_date, infer_datetime_format=True)
        end_date       = pd.to_datetime(end_date, infer_datetime_format=True) 
        dates          = pd.date_range(start_date, end_date) 

        results = []
        for date in dates[::launch_day_frequency]:
            # Calculate sunrise
            date = utils.calculate_sunrise(date, departure_point)

            vessel = Vessel.from_position(departure_point, 
                                          craft = craft,
                                          chart = chart,
                                          destination = destination,
                                          speed = speed,
                                          paddlers = paddlers,
                                          weight = weight,
                                          cadence = cadence,
                                          oar_depth = oar_depth,
                                          mode = mode,
                                          with_route = follows_route,
                                          params = vessel_params[mode][craft])


            # Interpolate the data for only the duration specified
            chart.interpolate(date, duration)

            # Use the interpolated values in the model
            model.use(chart)

            # Run the model
            vessel = model.run(vessel)

            start_date_str = date.strftime('%Y-%m-%dT%H:%M:%S')
            stop_date_str  = (date + pd.Timedelta(vessel.duration*timestep, unit='s')).strftime('%Y-%m-%dT%H:%M:%S')

            results.append(vessel.to_GeoJSON(start_date_str, stop_date_str, timestep))

        return results


    def run(self, chart = None, model = None, model_kwargs={}, chart_kwargs={}) -> Dict[str, Dict]:
        """Generates a set of trajectories in a date range, with a certain launch day frequency for the vessels.

        Args:
            chart (Chart, optional): the Chart instance to be used in this run. Defaults to None.
            model (Model, optional): the Model instance to be used in this run. Defaults to None.
            model_kwargs (dict, optional): Parameters for the model. Defaults to {}.
            chart_kwargs (dict, optional): Parameter for the chart. Defaults to {}.

        Returns:
            Dict[str, Dict]: A date-tagged dictionary with GeoJSON compliant dictionary results
        """

        # The chart object keeps track of the region of interest
        # and the wind/current data for that region
        # It is shared by all vessels
        if not chart:
            start_date = pd.to_datetime(date, infer_datetime_format=True)
            end_date   = start_date + pd.Timedelta(self.duration, unit='days')
            chart = Chart(self.bbox, start_date, end_date).load(self.data_directory, **chart_kwargs)
        
        # The model object describes the equations of movement and
        # traversal across the oceans over time
        if not model:
            model = Model(self.duration, self.timestep, **model_kwargs)

        results = {}
        for date in self.dates[::self.launch_day_frequency]:

            # Vessel objects are the individual agents traversing the ocean
            vessels = Vessel.from_positions(self.departure_points, 
                                            craft = self.craft,
                                            chart = chart, 
                                            destination = self.destination, 
                                            speed = self.speed, 
                                            mode = self.mode,
                                            with_route = self.follows_route,
                                            vessel_config=self.vessel_config)
            
            # Calculate sunrise
            date = utils.calculate_sunrise(date, self.departure_points[0])

            # Interpolate the data for only the duration specified
            chart.interpolate(date, self.duration)

            # Use the interpolated values in the model
            model.use(chart)

            trajectories = []

            for vessel in vessels:

                trajectories.append(model.run(vessel))

            # Add the trajectories for the date
            results.update({date.strftime('%Y-%m-%dT%H:%M:%S'): trajectories})

        return results


    def run_mp(self, model_kwargs={}, chart_kwargs={}) -> Dict[str, Dict]:
        """Pseudo-parallel generation of a set of trajectories in a date range, with a certain launch day frequency for the vessels.

        DEPR this function was never used and I don't expect to ever getting around to use it.

        Args:
            model_kwargs (dict, optional): Parameters for the model. Defaults to {}.
            chart_kwargs (dict, optional): Parameter for the chart. Defaults to {}.

        Returns:
            Dict[str, Dict]: A date-tagged dictionary with GeoJSON compliant dictionary results
        """
        # The chart object keeps track of the region of interest
        # and the wind/current data for that region
        # It is shared by all vessels
        chart = Chart(self.bbox, self.start_date, self.end_date).load(self.data_directory, **chart_kwargs)
        
        # The model object describes the equations of movement and
        # traversal across the oceans over time
        model = Model(self.duration, self.dt, **model_kwargs)

        results = {}
        for date in self.dates[::self.launch_day_frequency]:

            # Vessel objects are the individual agents traversing the ocean
            vessels = Vessel.from_positions(self.departure_points, 
                                            craft = self.craft,
                                            chart = chart, 
                                            destination = self.destination, 
                                            speed = self.speed, 
                                            mode = self.mode, 
                                            vessel_config=self.vessel_config)
            
            # Interpolate the data for only the duration specified
            chart.interpolate(date, self.duration)

            # Use the interpolated values in the model
            model.use(chart)

            with mp.Pool(mp.cpu_count()) as p:

                trajectories = p.map(model.run, vessels)

            # Add the trajectories for the date
            results.update({date.strftime('%Y-%m-%d'): trajectories})

        return results