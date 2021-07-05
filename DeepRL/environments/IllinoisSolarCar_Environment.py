import gym
from gym import spaces
from gym import wrappers

from math import sin, cos, pi
from dateutil import tz
import numpy as np
import datetime
import pytz

"""
CONSTANTS DEFINITIONS
"""
MAX_CAR_ACCELERATION = 2.5
MAX_CAR_DECELERATION = -1.5

STEPS_IN_EPISODE = 250  # How many steps in an episode (day).
MAX_SPEED = 35  # In m/s.
INITIAL_SPEED = 0
ACCELERATION = 2.5
DECELERATION = 1.5
STATE_SIZE = 6  # Number of things the agent can see from the environment.
DISCRETE = True

START_TIME = datetime.datetime.timestamp(
    datetime.datetime(2021, 7, 21, 8, 0, tzinfo=tz.tzoffset('CST', -5 * 3600)))  # 8:00 AM someday in july
END_TIME = datetime.datetime.timestamp(
    datetime.datetime(2021, 7, 21, 17, 0, tzinfo=tz.tzoffset('CST', -5 * 3600)))  # 5:00 PM that day

PACK_VOLTAGE = 100
BATTERY_CAPACITY = 5200 * 3600  # In Ws (Watt-second)
REGEN_EFFICIENCY = 0.1
MASS = 250
FRONTAL_AREA = 0.78
COEFFICIENT_OF_DRAG = 0.116
IDLE_POWER = 0.39
START_SOC = 1  # Choose an initial state of charge (percentage) for the battery at the beginning of the day.


class IllinoisSolarCar_Environment(gym.Env):
    metadata = {'render.modes': ['human']}
    environment_name = "Illinois Solar Car Environment"

    def __init__(self):
        super(IllinoisSolarCar_Environment, self).__init__()

        # Define action space.
        self.action_space = spaces.Discrete(3)

        # Define observation space.
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(6,)
        )

        # Random stuff from similar environments.
        self.reward_threshold = 0.0
        self.id = "Illinois Solar Car Environment"
        self.trials = 50


        # Finally, reset the environment.
        self.reset()

    def reset(self):
        # Keep track of the current step.
        self.current_step = 1

        # Starting speed.
        self.speed = INITIAL_SPEED

        # Get highest possible solar power at midday at Null island.
        self.highest_solar_power = self._get_solar_power(1601294400, latitude=0, longitude=0)
        # Get the maximum distance.
        self.max_possible_distance = MAX_SPEED * (END_TIME - START_TIME)

        # Get array of n evenly spaced times.
        self.times = np.linspace(START_TIME, END_TIME, STEPS_IN_EPISODE)
        # Find dt.
        self.tdiff = self.times[1] - self.times[0]

        # Calculate all solar powers through the day.
        self.solar_power = self._get_solar_power_array(self.times)

        # Keep track of the car's performance through the day.
        self.power_used = 0
        self.net_power = self.solar_power[0]
        self.net_energy = self.solar_power[0] * self.tdiff
        self.soc = START_SOC
        self.velocities = [0]
        self.distance_traveled = 0

        # Return initial observations.
        return self._next_observation()

    def _next_observation(self):
        observations = np.array([
            self.net_power / self.highest_solar_power,
            self.net_energy / (self.highest_solar_power * self.tdiff),
            self.distance_traveled / self.max_possible_distance,
            self.current_step / STEPS_IN_EPISODE,
            self.soc,
            self.velocities[-1] / MAX_SPEED,
        ], dtype=np.float32)

        return {"observation": observations,
            "desired_goal": np.array([647_000]),
            "achieved_goal": np.zeros([0]),
        }

    def _take_action(self, action):
        # Change speed based on action.
        self.speed += ACCELERATION if action == 2 else (-DECELERATION) if action == 0 else 0
        self.speed = min(max(self.speed, 0), MAX_SPEED)

    def step(self, action):
        # Take the step in the environment.
        self._take_action(action)

        # Record new datapoints.
        self.velocities.append(self.speed)
        self.power_used = self._get_power_usage(self.speed)
        self.net_power = self.solar_power[self.current_step] - self.power_used
        self.net_energy = self.net_power * self.tdiff + self.net_energy
        self.soc = min((self.net_energy / BATTERY_CAPACITY) + START_SOC, 1)
        distanceTraveledInThisStep = self.speed * self.tdiff
        self.distance_traveled += distanceTraveledInThisStep

        # Calculate reward.
        reward = distanceTraveledInThisStep

        # Calculate if done.
        done = (self.current_step >= STEPS_IN_EPISODE) or (self.soc <= 0)

        # Generate next observation.
        observation = self._next_observation()

        # Increment current step.
        self.current_step += 1

        return observation, reward, done, {}

    def render(self, mode='human', close=False):
        print(f"Step: {self.current_step}/{STEPS_IN_EPISODE}")
        print(f"SOC: {self.soc}")
        print(f"Dist. Traveled: {self.distance_traveled} m")
        print(f"Speed: {self.speed} m/s")

    """ ----------------------------------------------------------------------------------------------------
                                        Helper functions.
        ----------------------------------------------------------------------------------------------------
    """

    def _get_solar_power(self, time_stamp, latitude=40.2139, longitude=-88.2434, cloud_cover=0):
        """_get_solar_power - function that gets the solar power recovered by the car at a certain time

        @arg time_stamp - unix timestamp for the time you want the solar power
        @arg latitude, longitude - geoposition to get solar power (default is Champaign)
        @arg cloud_cover - weather condition for how covered the sun is by clouds
        """
        race_timezone_offset = -5

        time_obj = datetime.datetime.utcfromtimestamp(
            time_stamp + (race_timezone_offset * 3600)
        ).timetuple()

        panel_tilt = 0

        panel_azimuth_angle = 0
        day_of_year = time_obj.tm_yday
        Latitude = latitude * (2 * np.pi / 360)

        B = (day_of_year - 1 + ((time_obj.tm_hour - 12) / 24)) * (2 * np.pi / 366.)
        E = 229.18 * (0.000075 + 0.001868 * cos(B) - 0.032077 * sin(B) - 0.014615 * cos(2 * B) - 0.040849 * sin(2 * B))
        time_correction_factor = E + 4 * longitude - (60 * race_timezone_offset)
        local_solar_time = time_obj.tm_hour * 60 + time_obj.tm_min + time_correction_factor

        Solar_Declination = (-23.45 * cos((day_of_year + 10) * 2 * pi / 365)) * (2 * np.pi / 360)
        Apparent_Solar_Irradiance = 1160 + 75 * sin((day_of_year - 275) * 2 * pi / 365)

        Optical_Depth = 0.174 + 0.035 * sin((day_of_year - 100) * 2 * pi / 366)
        Solar_Hour_Angle = np.radians((local_solar_time / 4) - 180)
        Solar_Zenith_Angle = np.arccos(np.sin(Latitude) * np.sin(Solar_Declination)
                                       + np.cos(Latitude) * np.cos(Solar_Declination) * np.cos(Solar_Hour_Angle))
        Solar_Altitude_Angle = (np.pi / 2) - Solar_Zenith_Angle

        theta = 90 - (Solar_Altitude_Angle * 180 / np.pi)
        Air_Mass_Ratio = 1 / cos(np.pi / 2 - Solar_Altitude_Angle)
        Clear_Sky_Direct_Beam_Radiation = Apparent_Solar_Irradiance * np.exp(- Optical_Depth * Air_Mass_Ratio)

        Beam_Panel_Incidence_Angle = Solar_Altitude_Angle

        Panel_Irradiation = Clear_Sky_Direct_Beam_Radiation * sin(Beam_Panel_Incidence_Angle)
        Incident_Solar_Power = Panel_Irradiation

        return .7 * max(0, Incident_Solar_Power * .225 * 4) * (1 - cloud_cover)

    def _get_solar_power_array(self, times):
        return np.vectorize(self._get_solar_power)(times)

    def _get_power_usage(self, velocity, slope=0):
        """get power usage based on the velocity and the slope

        @arg velocity in m/s
        @arg slope in %
        """
        rads = np.arctan(slope / 100)

        aero_loss = ((velocity) ** 2 * 0.5 * 1.225 * FRONTAL_AREA * COEFFICIENT_OF_DRAG) * velocity
        rolling_loss = .8 / (.588 / 2) * 4 * velocity
        hill_loss = (9.81 * MASS * velocity * np.sin(rads))
        if (hill_loss < 0):
            hill_loss *= REGEN_EFFICIENCY

        return 1.3 * (aero_loss + rolling_loss + hill_loss) + (IDLE_POWER * PACK_VOLTAGE)