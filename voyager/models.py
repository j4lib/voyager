from typing import Tuple
import numpy as np
import pandas as pd
import xoak
import xarray as xr

from .vessel import Vessel
from .chart import Chart
from .move import Displacement
from .utils import twilight_type, calculate_sunrise, is_it_night, calculate_twilights

class Model:
    """A class to represent the model being simulated. A "model" in this context represents the set of modelling choices.

    Attributes:
        duration (int): maximal duration of each travel in days
        dt (float): time step of the simulation in seconds
        chart (Chart): Chart over which the simulation takes place.
        sigma (float): incertitude of displacement (as sqrt of variance)
        angle_sigma (float): incertitude of direction towards target (as sqrt of variance)
        tolerance (float): how close to a target is considered enough to end the simulation
    Methods:
        use(chart): use a supplied chart object of winds and currents
        velocity(t, longitude, latitude): extracts currents and winds at a specific point in space and time
        run(vessel): Calculates the trajectory of a vessel object in space over time
    """

    def __init__(self, duration: int, dt: float, sigma = 2000.0, angle_sigma = 10.0, tolerance = 0.5e-3) -> None:
        """
        Args:
            duration (int): maximal duration of each travel in days
            dt (float): time step of the simulation in seconds
            sigma (float): incertitude of displacement (as sqrt of variance). Defaults to 2000.0.
            angle_sigma (float): incertitude of direction towards target (as sqrt of variance). Defaults to 10 (degrees).
            tolerance (float): how close to a target is considered enough to end the simulation. Defaults to 0.5e-3.
        """
        self.duration = duration
        self.dt       = dt
        self.chart = None
        self.sigma = sigma
        self.angle_sigma = angle_sigma
        self.tolerance = tolerance

    def use(self, chart: Chart):
        """Use a supplied chart object of winds and currents.

        Args:
            chart (Chart): A Chart object with winds and currents

        Returns:
            Model: The Model instance (self)
        """

        self.chart = chart

        return self

    def velocity(self, t: float, longitude: float, latitude: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate a tuple of (current, wind) velocities at a specific time and set of
        WGS84 coordinates through interpolation.

        Args:
            t (float): time
            longitude (float): Longitude (WGS84)
            latitude (float): Latitude (WGS84)

        Returns:
            Tuple[np.ndarray, np.ndarray]: The current and wind velocities respectively
        """

        assert self.chart != None

        # Calculate current speeds
        v_x_current = self.chart.u_current((t, longitude, latitude))
        v_y_current = self.chart.v_current((t, longitude, latitude))

        # Test if we have reached land. If so, break simulation
        # TODO write a function that does not always stops the simulation when land is hit.
        if np.isnan(v_x_current) or np.isnan(v_y_current):
            return None, None

        # Calculate wind speeds
        v_x_wind = self.chart.u_wind((t, longitude, latitude))
        v_y_wind = self.chart.v_wind((t, longitude, latitude))

        return (np.array([v_x_current, v_y_current]), np.array([v_x_wind, v_y_wind]))
    
    def real_velocity(self, t: float, longitude: float, latitude: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate a tuple of (current, wind) velocities at a specific time and set of
        WGS84 coordinates without interpolation. This function is optimized for our current datasets.

        Args:
            t (float): time
            longitude (float): Longitude (WGS84)
            latitude (float): Latitude (WGS84)

        Returns:
            Tuple[float, float]: The current and wind velocities respectively
        """

        assert self.chart != None

        N_SECONDS_IN_DAY = 86400
        current_time = self.chart.start_date + pd.Timedelta(t*N_SECONDS_IN_DAY, unit="seconds")

        # Calculate current speeds
        v_x_current = self.chart.u_current_all.sel(time=current_time, 
                                                    longitude=longitude, 
                                                    latitude=latitude,
                                                    method="nearest")
        v_y_current = self.chart.v_current_all.sel(time=current_time, 
                                                    longitude=longitude, 
                                                    latitude=latitude,
                                                    method="nearest")

        # Test if we have reached land. If so, break simulation
        # TODO write a function that does not always stops the simulation when land is hit.
        if np.isnan(v_x_current) or np.isnan(v_y_current):
            return None, None

        # Wind data are set with multiple coordinates, where lons / lats are not coordinates, but values inscripted on a system of
        # logical (x,y)-coordinates. So we use the package xoak to find the cell closest to coordinates lon/lat as given in the input (see
        # xoak Documentation for how this is set up).
        #self.chart.u_wind_all.xoak.set_index(['longitude', 'latitude'], index_type='scipy_kdtree')
        u_selected_grid_cell = self.chart.u_wind_all.xoak.sel(longitude=xr.DataArray([longitude, ]), latitude=xr.DataArray([latitude,]))
        v_x_wind = float(u_selected_grid_cell.sel(time=current_time, method="nearest").values)

        #self.chart.v_wind_all.xoak.set_index(['longitude', 'latitude'], index_type='scipy_kdtree')
        v_selected_grid_cell = self.chart.v_wind_all.xoak.sel(longitude=xr.DataArray([longitude, ]), latitude=xr.DataArray([latitude,]))
        v_y_wind = float(v_selected_grid_cell.sel(time=current_time, method="nearest").values)

        return (np.array([v_x_current, v_y_current]), np.array([v_x_wind, v_y_wind]))

    def wave_height(self, t: float, longitude: float, latitude: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts wave height at a specific time and set of WGS84 coordinates without interpolation. 
        This function is optimized for our current datasets.

        Args:
            t (float): time
            longitude (float): Longitude (WGS84)
            latitude (float): Latitude (WGS84)

        Returns:
            float: The current wave height
        """
        assert self.chart != None

        N_SECONDS_IN_DAY = 86400
        current_time = self.chart.start_date + pd.Timedelta(t*N_SECONDS_IN_DAY, unit="seconds")

        # Calculate current speeds
        wave_height = float(self.chart.waves_all.sel(time=current_time, 
                                                     longitude=longitude, 
                                                     latitude=latitude,
                                                     method="nearest"))

        return wave_height


    def run(self, vessel: Vessel) -> Vessel:
        """Calculates the trajectory of a vessel object in space over time.

        Trajectories are calculated by estimating the displacement in km from a position in
        (longitude, latitude) coordinates, and converting the displacement back to (longitude, latitude).

        Simulation is stopped when the vessel encounters NaN, indicating land or area outside the simulation region.

        Assumes a spherical Earth.

        Args:
            vessel (Vessel): Vessel object with initial position

        Returns:
            Vessel: A modified vessel object with full trajectory
        """

        # Set random seed
        # Important, otherwise all virtual threads will return the same result
        np.random.seed()

        longitude = vessel.x
        latitude  = vessel.y

        # Constant
        N_SECONDS_IN_DAY = 86400

        # Initialization
        dx = 0
        dy = 0

        target_tol = (self.dt) * self.tolerance # 1/1000 is a good value

        # The type of displacement is handled by the vessel mode of traversal
        displacement = Displacement(vessel, self.dt)

        for t in np.arange(start=0, stop=self.duration, step=self.dt/N_SECONDS_IN_DAY):
            
            # Calculate interpolated velocity at current coordinates
            c, w = self.velocity(t, longitude, latitude)
            waves = self.wave_height(t, longitude, latitude)

            # if c, w, None: either it's land, or interpolate has boundaries issues, 
            # so just test velocity on the real data before determining whether it's land.
            if c is None or w is None:
                c, w = self.real_velocity(t, longitude, latitude)

            # If c, w still None, then land has been reached!
            if c is None or w is None:
                break

            # Calculate displacement
            dx, dy = displacement.move(c, w, self.angle_sigma)\
                                 .with_uncertainty(sigma=self.sigma)\
                                 .km()
               
            # print(dx, dy)
            # Calculate new longitude, latitude from displacement
            # Using great circle distances
            longitude, latitude = displacement.to_lonlat(dx, dy, longitude, latitude)

            # Update vessel data
            vessel.update_distance(dx, dy)\
                  .update_position(longitude, latitude)\
                  .update_mean_speed(self.dt)\
                  .update_encountered_environment(c, w, waves)

            # Check progress along route
            is_arrived = vessel.has_arrived(longitude, latitude, target_tol)

            if is_arrived:
                break

        return vessel

    def run_by_day(self, vessel: Vessel) -> Tuple[Vessel, bool]:
        """Calculates the trajectory of a vessel object in space over time, stops trip at night.

        Trajectories are calculated by estimating the displacement in km from a position in
        (longitude, latitude) coordinates, and converting the displacement back to (longitude, latitude).

        Simulation is stopped when the vessel encounters NaN, indicating land or area outside the simulation region.

        Assumes a spherical Earth.

        Args:
            vessel (Vessel): Vessel object with initial position

        Returns:
            Vessel: A modified vessel object with full trajectory
        """

        # Set random seed
        # Important, otherwise all virtual threads will return the same result
        np.random.seed()

        longitude = vessel.x
        latitude  = vessel.y

        # Constant
        N_SECONDS_IN_DAY = 86400

        # Initialization
        dx = 0
        dy = 0
        initial_day_time, final_day_time = calculate_twilights(vessel.launch_date.date(), [longitude, latitude], twilight_type)
        is_night = False

        target_tol = (self.dt) * self.tolerance # 1/1000 is a good value

        # The type of displacement is handled by the vessel mode of traversal
        displacement = Displacement(vessel, self.dt)
        
        for t in np.arange(start=0, stop=self.duration, step=self.dt/N_SECONDS_IN_DAY):
            current_day_time = initial_day_time + pd.Timedelta(t, unit="days")
            
            # record whether last night was night (used as check)
            last_night = is_night
            # measure whether now is night
            is_night = is_it_night(current_day_time, [longitude, latitude], twilight_type)

            if is_night:
                # if it's night, check whether last night was also night. If not, update stops. If it was, just continue.
                if not last_night:
                    vessel.update_stops(current_day_time, [longitude, latitude])
                
                vessel.update_distance(0.0, 0.0)\
                  .update_position(longitude, latitude)\
                  .update_mean_speed(self.dt)\
                  .update_encountered_environment(c, w, waves)

                continue
            
            # if not is_night, then it will calculate new positions

            # Calculate interpolated velocity at current coordinates
            c, w = self.velocity(t, longitude, latitude)
            waves = self.wave_height(t, longitude, latitude)

            # if c, w, None: either it's land, or interpolate has boundaries issues, 
            # so just test velocity on the real data before determining whether it's land.
            if c is None or w is None:
                c, w = self.real_velocity(t, longitude, latitude)

            # If c, w still None, then land has been reached!
            if c is None or w is None:
                break

            # Calculate displacement
            dx, dy = displacement.move(c, w, self.angle_sigma)\
                                 .with_uncertainty(sigma=self.sigma)\
                                 .km()
               
            # Calculate new longitude, latitude from displacement
            # Using great circle distances
            longitude, latitude = displacement.to_lonlat(dx, dy, longitude, latitude)

            # Update vessel data
            vessel.update_distance(dx, dy)\
                  .update_position(longitude, latitude)\
                  .update_mean_speed(self.dt)\
                  .update_encountered_environment(c, w, waves)

            # Check progress along route
            is_arrived = vessel.has_arrived(longitude, latitude, target_tol)

            if is_arrived:
                break

        return vessel