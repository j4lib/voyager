from . import geo
from .utils import angle_uncertainty
import numpy as np
import pandas as pd
import math
from typing import *

class Displacement:
    """A class used to represent a displacement on a Chart object.

    Attributes:
        vessel (Vessel): the Vessel object that is displaced
        dt (float): the time step between displacements
        dxy Tuple[float, float]: displacement in km on the x and y axis
    Methods:
        move(c, w, angle_sigma): Creates the displacement due to current and wind velocity
        leeway_displacement(w, Sl, Yt, dt): calculates the leeway displacement from vessel parameters and wind speed
        leeway_velocity(w, Sl, Yt): calculates the leeway wind velocity from vessel parameters and wind speed
        levison_leeway_displacement(w, dt): calculates the displacement due to leeway forces, using the Levison method.
        rotate(x, angle): rotates an array x by a certain angle
        from_drift(self, c: np.ndarray, w: np.ndarray): Generate displacement due to only drifting with the winds and currents.
        from_paddling(c, w, angle_sigma, position, target, speed): Generate displacement due to paddling with a certain paddling speed, as well as environmental factors from currents and winds.
        from_sailing(c, w, angle_sigma, position, target): Generate displacement due to sailing, reinforcing the wind speed contribution over the currents.
        knots_to_si(knots): Converts knots to SI units (m/s)
        si_to_knots(si): Converts SI units (m/s) to knots
        paddling_speed(w, bearing): Calculates the paddling speed according to Wolfson Unit diagrams (in m/s)
        paddling_leeway(w, bearing): Calculates the paddling speed according to Wolfson Unit diagrams (in m/s)
        with_uncertainty(sigma=1): Adds normal distributed noise to the current position.
        km(): Returns the displacement in kilometres, from metres.
        to_lonlat(dx, dy, longitude, latitude): Convenience function to convert a displacement into longitude and latitude

    """
    def __init__(self, vessel, dt) -> None:
        """
        Args:
            vessel (Vessel): the Vessel object that is displaced
            dt (float): the time step between displacements
            dxy (np.array): the displacement at current timestep, with two elements x and y
        """
        super().__init__()

        self.vessel = vessel
        self.dt     = dt
        self.dxy = None

    def move(self, c: np.ndarray, w: np.ndarray, landmarks: Tuple[float, float], angle_sigma: float):
        """Creates the displacement due to a current and wind velocity.

        Args:
            c (np.ndarray): current velocity
            w (np.ndarray): wind velocity
            landmarks (Tuple[float, float]): distance (in km) and angle (in degrees from the North) from the closest land
            angle_sigma(float): angle variance (error) in choice of bearing

        Raises:
            ValueError: Raised if the model displacement is not drifting, paddling or sailing

        Returns:
            Displacement: The Displacement instance
        """

        if self.vessel.mode == 'drifting':
            return self.from_drift(c, w)

        elif self.vessel.mode == 'paddling':
            return self.from_paddling(c, w, angle_sigma, landmarks, (self.vessel.x, self.vessel.y), self.vessel.target, self.vessel.speed)

        elif self.vessel.mode == 'sailing':
            return self.from_sailing(c, w, angle_sigma, (self.vessel.x, self.vessel.y), self.vessel.target)

        else:
            raise ValueError("Mode of displacement should be drifting, paddling or sailing")

    @staticmethod
    def leeway_displacement(w: np.ndarray, Sl: float, Yt: float, dt: float):
        """Calculates the leeway displacement from vessel parameters 
        and wind speed.

        Args:
            w (np.ndarray): The wind velocity with two components
            Sl (float): vessel parameter
            Yt (float): vessel parameter
            dt (float): Time step

        Returns:
            np.ndarray: The displacement due to leeway wind
        """

        # Calculate leeway velocity from wind
        leeway = Displacement.leeway_velocity(w, Sl, Yt)

        # Convert leeway to m/s from knots
        leeway = Displacement.knots_to_si(leeway)

        # Calculate displacement
        dxy_leeway = leeway * dt

        return dxy_leeway

    @staticmethod
    def leeway_velocity(w: np.ndarray, Sl: float, Yt: float) -> np.ndarray:
        """Calculates the leeway wind velocity from vessel parameters and
        wind speed.

        Args:
            w (np.ndarray): The wind velocity with two components
            Sl (float): vessel parameter
            Yt (float): vessel parameter

        Returns:
            np.ndarray: The velocity from leeway wind
        """

        # Convert from m/s to knots
        w = Displacement.si_to_knots(w)

        leeway = np.zeros_like(w)

        if np.abs(w[0]) > 6:
            leeway[0] = (Sl * w[0]) + Yt
        else:
            leeway[0] = (Sl + Yt / 6) * w[0]

        if np.abs(w[1]) > 6:
            leeway[1] = (Sl * w[1]) + Yt
        else:
            leeway[1] = (Sl + Yt / 6) * w[1]

        return leeway

    @staticmethod
    def levison_leeway_displacement(w: np.ndarray, dt: float):
        """Calculates the displacement due to leeway forces,
        using the Levison method.

        Args:
            w (np.ndarray): Wind velocity
            dt (float): Timestep

        Raises:
            ValueError: Raised if the absolute wind velocity is a non-positive number

        Returns:
            np.ndarray: The displacement due to leeway forces
        """

        # Convert from m/s to knots
        w = Displacement.si_to_knots(w)

        # Prefill the leeway velocity
        leeway = np.zeros_like(w)

        for idx in (0, 1):

            w_abs = np.abs(w[idx])
            w_sign = np.sign(w[idx])
            
            if w_abs < 1:
                leeway[idx] = 0
            elif 1 <= w_abs <= 3:
                leeway[idx] = 0.5
            elif 3 < w_abs <= 6:
                leeway[idx] = 1 
            elif 6 < w_abs <= 10:
                leeway[idx] = 2
            elif 10 < w_abs <= 16:
                leeway[idx] = 3 
            elif 16 < w_abs <= 21:
                leeway[idx] = 4.5
            elif 21 < w_abs <= 27:
                leeway[idx] = 6 
            elif 27 < w_abs <= 33:
                leeway[idx] = 7 
            elif 33 < w_abs <= 40:
                leeway[idx] = 6 
            elif w_abs > 40:
                leeway[idx] = 4.5
            else:
                raise ValueError(f"Invalid absolute velocity {w_abs}")

            leeway[idx] *= w_sign

        leeway = Displacement.knots_to_si(leeway)

        dxy_leeway = leeway * dt

        return dxy_leeway


    @staticmethod
    def rotate(x: np.ndarray, angle: float) -> np.ndarray:
        """Rotates an array x by a certain angle.

        Args: 
            x: array to rotate
            angle: angle of rotation
        Returns:
            np.ndarray: rotate array
        """

        r = np.array(( 
                    (np.cos(angle), -np.sin(angle)),
                    (np.sin(angle),  np.cos(angle)) 
                    ))

        return r.dot(x)

    def from_drift(self, c: np.ndarray, w: np.ndarray):
        """Generate displacement due to only drifting with the winds and currents. 

        Args:
            c (np.ndarray): Current velocity
            w (np.ndarray): Wind velocity

        Returns:
            Displacement: The Displacement instance
        """

        # Calculate the drift due to the currents
        dxy_current = c * self.dt

        self.dxy = dxy_current

        if self.vessel.craft != 7 and self.vessel.craft != 'hjortspring':

            # Load vessel parameters
            Sl = self.vessel.params["Sl"]
            Yt = self.vessel.params["Yt"]
            Da = self.vessel.params["Da"]

            # Convert to degrees
            Da = np.deg2rad(Da)

            # the deflections due to Da half right
            ## and half left of the wind
            flip = np.random.choice((1, -1))

            # Calculate the leeway speed and displacement
            dxy_leeway = Displacement.leeway_displacement(w, Sl, Yt, self.dt)

            # Calculate the deflection as a rotation
            dxy_deflect = Displacement.rotate(dxy_leeway, angle=Da*flip)

            # Total displacement in metres
            self.dxy = dxy_current + dxy_deflect 

        elif self.vessel.craft == 7:

            dxy_leeway = Displacement.levison_leeway_displacement(w, self.dt)

            self.dxy = dxy_leeway + dxy_current

        elif self.vessel.craft == 'hjortspring':
            pass
            

        return self

    def from_paddling(self, c: np.ndarray, w: np.ndarray, angle_sigma: float, landmarks: Tuple[float, float], position: np.ndarray, target: np.ndarray, speed: float):
        """Generate displacement due to paddling with a certain paddling speed, as well as environmental factors from
        currents and winds.

        Args:
            c (np.ndarray): Current velocity
            w (np.ndarray): Wind velocity
            angle_sigma (float): sigma of normally distributed angle error
            landmarks (Tuple[float, float]): distance (in km) and angle (in degrees from the North) of the closest land
            position (np.ndarray): Current position coordinates
            target (np.ndarray): Destination position coordinates
            speed (float): Paddling speed

        Returns:
            Displacement: The Displacement instance
        """

        # Calculate the bearing from the current position to the target
        a = geo.bearing_from_lonlat(position, target)
        a = np.deg2rad(a + angle_uncertainty(angle_sigma))

        # if landmarks are None, no land is found in an angle of 0.5 degrees around the coordinates. Bearing can be the same. 
        # If there is land in this angle, we check for land within 5 km, then stir away if there is.
        if landmarks[0] is None:
            pass
        else:
            # calculate whether land is ahead. We define "ahead" as within an angle of 90 degrees around the bearing:
            land_angle = np.deg2rad(landmarks[1])
            left = (a - np.pi/4 + 2*np.pi) % (2*np.pi)
            right = (a + np.pi/4 + 2*np.pi) % (2*np.pi)
            
            if np.pi/4 <= a <= 7/4*np.pi:
                is_ahead = left <= land_angle <= right
            else:
                is_ahead = (0 <= land_angle <= right) or (left <= land_angle <= 2*np.pi)
            # elif a > 7/4*np.pi:
            #     is_ahead = (left <= land_angle <= 2*np.pi) or (0 <= land_angle <= right)

            if is_ahead:
                # in general, if (bearing - land_angle) > 0 , then land is on the left (steering to the right - positive - is necessary). And viceversa.
                sign_of_steering = np.sign(a - land_angle)
                if sign_of_steering != 0:
                    a = a - sign_of_steering * 3*np.pi/2
                else:
                    # the case where (bearing - land_angle) = 0 represent land right ahead. In this (hopefully rare) case we just move in the 
                    # opposite direction and try again...
                    a = a + np.pi
                
            else:
                pass

        # Get the displacement due to paddling towards the target
        if self.vessel.craft == 'hjortspring':
            real_direction = self.paddling_leeway(w, a)

            # the calculation of the paddling speed is done with comparison to the bearing, but movement is in the real_direction
            dxy_paddle = self.paddling_speed(w, a) * self.dt * np.array([-np.sin(real_direction), np.cos(real_direction)])
        else:
            dxy_paddle = speed * self.dt * np.array([-np.sin(a), np.cos(a)])

        # Calculate the displacement due to drift
        dxy_drift = self.from_drift(c, w).dxy

        self.dxy = dxy_drift + dxy_paddle

        return self

    def from_sailing(self, c: np.ndarray, w: np.ndarray, angle_sigma: float, position: np.ndarray, target: np.ndarray):
        """Generate displacement due to sailing, reinforcing the wind speed contribution over the currents.

        Args:
            c (np.ndarray): Current velocity
            w (np.ndarray): Wind velocity
            position (np.ndarray): Current position coordinates
            target (np.ndarray): Destination position coordinates

        Raises:
            ValueError: Raised if the angle between the bearing and reference is a non-positive number

        Returns:
            Displacement: The Displacement instance
        """

        position = np.array(position)
        target   = np.array(target)

        # Calculate the drift due to the currents
        dxy_c = c * self.dt

        # Calculate the bearing
        bearing = target - position
        a = geo.bearing_from_lonlat(position, target)
        a = np.deg2rad(a + angle_uncertainty(angle_sigma))
        bearing = np.array([np.cos(a), np.sin(a)]).squeeze()

        bearing = bearing.squeeze()
        w   = w.squeeze()

        # Angle between bearing and reference vector
        b = np.arctan2(np.linalg.det([bearing, w]), np.dot(bearing, w))
        b = np.abs(np.rad2deg(b))

        w_abs = np.linalg.norm(w)
        
        mt          = self.vessel.params["mt"]
        wf_0_40     = self.vessel.params["wf 0-40"]
        wf_40_80    = self.vessel.params["wf 40-80"]
        wf_80_100   = self.vessel.params["wf 80-100"]
        wf_100_110  = self.vessel.params["wf 100-110"]
        wf_110_120  = self.vessel.params["wf 110-120"]

        if b  <= 40:
            sailing_velocity = wf_0_40 
        elif b > 40 or b <= 80:
            sailing_velocity = wf_40_80 
        elif b > 80 or b <= 100:
            sailing_velocity = wf_80_100 
        elif b > 100 or b <= 110:
            sailing_velocity = wf_100_110 
        elif b > 110:
            sailing_velocity = wf_110_120 
        else:
            raise ValueError(f"Invalid angle b={b}")

        sailing_velocity *= w_abs

        if b <= mt:
            displacement = sailing_velocity * self.dt
        else:
            tacking = np.deg2rad(b-mt)
            displacement = np.cos(tacking)*sailing_velocity*self.dt

        dxy_sailing = displacement * np.array([-np.sin(a), np.cos(a)])


        self.dxy = dxy_sailing + dxy_c

        return self

    @staticmethod
    def knots_to_si(knots: float) -> float:
        """Converts knots to SI units (m/s)

        Args:
            knots (float): Speed in knots

        Returns:
            float: Speed in metres/second
        """

        return knots / 1.94

    @staticmethod
    def si_to_knots(si: float) -> float:
        """Converts SI units (m/s) to knots

        Args:
            si (float): Speed in m/s

        Returns:
            float: Speed in knots
        """

        return si * 1.94

    def paddling_speed(self, w: np.ndarray, bearing: np.float64):
        """Calculates the paddling speed according to Wolfson Unit diagrams (in m/s), when existing.

        Args:
            w (np.ndarray): Wind velocity (two components)
            bearig (np.float64): bearing of the boat (radiants)

        Raises:
            ValueError: Raised if the angle between the bearing and reference is a non-positive number
            ValueError: Raised if the speed of the wind is higher than 30
        """

        try:
            file_polar_diagram = f"./voyager/configs/hjortspring_speeds_{self.vessel.paddlers}pad_{self.vessel.weight}kg_{self.vessel.cadence}cad_{self.vessel.oar_depth}oars.txt"
            polar_diagram = pd.read_csv(file_polar_diagram, sep="\t", index_col=0)
        except:
            raise ValueError(f"The file corresponding to this vessel was not found!")
        
        # angle between bearing and wind
        bearing_decomposed = np.array([np.cos(bearing), np.sin(bearing)]).squeeze()
        bearing_decomposed = bearing_decomposed.squeeze()
        w = w.squeeze()

        true_wind_angle = np.arctan2(np.linalg.det([bearing_decomposed, w]), np.dot(bearing_decomposed, w))
        true_wind_angle = np.abs(np.rad2deg(true_wind_angle))

        true_wind_speed = np.linalg.norm(w)
        true_wind_speed_knots = Displacement.si_to_knots(true_wind_speed)

        # round angle to next 10 and speed to next 5
        if 0 <= true_wind_angle <= 180:
            rounded_angle = math.ceil(true_wind_angle/10)*10
        else:
            raise ValueError(f"Wind angle is not between 0 and 180 ({true_wind_angle} deg)")
        if 0 <= true_wind_speed_knots <= 30:
            rounded_speed = math.ceil(true_wind_speed_knots/5)*5
        elif true_wind_speed_knots > 30:
            # OPEN what if speed is too high? Set final speed of boat to zero, possibly
            rounded_speed = 30
        else:
            raise ValueError(f"Wind speed is negative ({true_wind_speed} m/s)")

        speed_in_knots = polar_diagram[str(rounded_speed)][rounded_angle]
        speed = Displacement.knots_to_si(speed_in_knots)

        return speed


    def paddling_leeway(self, w: np.ndarray, bearing: np.float64):
        """Calculates the paddling speed according to Wolfson Unit diagrams (in m/s), when existing.

        Args:
            w (np.ndarray): Wind velocity (two components)
            bearing (np.float64): bearing of the boat (radiants)

        Raises:
            ValueError: Raised if the angle between the bearing and reference is a non-positive number
            ValueError: Raised if the speed of the wind is higher than 30
        """

        try:
            file_polar_diagram = f"./voyager/configs/hjortspring_leeway_{self.vessel.paddlers}pad_{self.vessel.weight}kg_{self.vessel.cadence}cad_{self.vessel.oar_depth}oars.txt"
            polar_diagram = pd.read_csv(file_polar_diagram, sep="\t", index_col=0)
        except:
            raise ValueError(f"The file corresponding to this vessel was not found!")
        
        # angle between bearing and wind
        bearing_decomposed = np.array([np.cos(bearing), np.sin(bearing)]).squeeze()
        bearing_decomposed = bearing_decomposed.squeeze()
        w = w.squeeze()

        true_wind_angle = np.arctan2(np.linalg.det([bearing_decomposed, w]), np.dot(bearing_decomposed, w))

        # find the wind_sign, will be used in leeway_angle to apply the correct angle to leeway.
        wind_sign = np.sign(true_wind_angle)
        # # then just
        true_wind_angle = np.abs(np.rad2deg(true_wind_angle))

        true_wind_speed = np.linalg.norm(w)
        true_wind_speed_knots = Displacement.si_to_knots(true_wind_speed)

        # round angle to next 10 and speed to next 5
        if 0 <= true_wind_angle <= 180:
            rounded_angle = math.ceil(true_wind_angle/10)*10
        else:
            raise ValueError(f"Wind angle is not between 0 and 180 ({true_wind_angle} deg)")
        if 0 <= true_wind_speed_knots <= 30:
            rounded_speed = math.ceil(true_wind_speed_knots/5)*5
        elif true_wind_speed_knots > 30:
            # OPEN what if speed is too high? Set final speed of boat to zero, possibly
            rounded_speed = 30
        else:
            raise ValueError(f"Wind speed is negative ({true_wind_speed} m/s)")

        leeway_angle = -wind_sign*polar_diagram[str(rounded_speed)][rounded_angle]

        new_bearing = np.deg2rad(leeway_angle) + bearing

        return new_bearing


    def with_uncertainty(self, sigma=1) -> np.ndarray:
        """Adds normal distributed noise to the current position.

        Args:
            sigma (float): The standard deviation of the added noise. Default: 1.

        Returns:
            Displacement: The Displacement object
        """

        self.dxy += np.random.normal(0, sigma, size=self.dxy.shape)
        
        return self


    def km(self):
        """Returns the displacement in kilometres, from metres.

        Returns:
            np.array: A numpy array with shape (2,) with the displacement in kilometres.
        """
        
        return self.dxy / 1e3

    def to_lonlat(self, dx: float, dy: float, longitude: float, latitude: float) -> Tuple[float, float]:
        """Convenience function to convert a displacement into longitude and latitude

        Args:
            dx (float): Displacement in x-axis
            dy (float): Displacement in y-axis
            longitude (float): Longitudinal displacement
            latitude (float): Latitudinal displacement

        Returns:
            Tuple[float, float]: A tuple of the longitude and latitude
        """

        return geo.lonlat_from_displacement(dx, dy, (longitude, latitude))
