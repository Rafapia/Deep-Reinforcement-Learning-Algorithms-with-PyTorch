import gym
from gym import spaces
from gym.utils import seeding

from math import sin, cos, pi
from dateutil import tz
import numpy as np
import datetime

config = {

    "STEPS_IN_EPISODE": 250, # How many steps in an episode (day).
    "MAX_SPEED": 35, # In m/s.
    "INITIAL_SPEED": 0,
    "ACCELERATION": 2.5,
    "DECELERATION": 1.5,
    "STATE_SIZE": 6, # Number of things the agent can see from the environment.
    "ACTION_SIZE": 3,

    "START_TIME": datetime.datetime.timestamp(datetime.datetime(2021, 7, 21, 8, 0, tzinfo=tz.tzoffset('CST', -5*3600))), # 8:00 AM someday in july
    "END_TIME": datetime.datetime.timestamp(datetime.datetime(2021, 7, 21, 17, 0, tzinfo=tz.tzoffset('CST', -5*3600))), # 5:00 PM that day

    "PACK_VOLTAGE": 100,
    "BATTERY_CAPACITY": 5200 * 3600, # In Ws (Watt-second)
    "REGENERATIVE_BREAK_EFFICIENCY": 0.1,
    "MASS": 250,
    "FRONTAL_AREA": 0.78,
    "COEFFICIENT_OF_DRAG": 0.116,
    "IDLE_POWER": 0.39,
    "START_SOC": 1, # Choose an initial state of charge (percentage) for the battery at the beginning of the day.
}


class SimpleISC(gym.Env):
    metadata = {'render.mode': ['human']}

    def __init__(self):
        # Gym requirement.
        self.action_space = spaces.Discrete(config["ACTION_SIZE"])
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(config["STATE_SIZE"],)
        )
        self.reward_threshold = 0.0
        self.trials = 10

        # Setting all constants.
        self.speed = config["INITIAL_SPEED"]

        self.HIGHEST_SOLAR_POWER = self._get_solar_power(1601294400, latitude=0, longitude=0)
        self.MAX_DISTANCE = config["MAX_SPEED"] * (config["END_TIME"] - config["START_TIME"])
        self.reward_range = np.array([0, self.MAX_DISTANCE])

        self.TIMES = np.linspace(config["START_TIME"], config["END_TIME"], config["STEPS_IN_EPISODE"])
        self.TDIFF = self.TIMES[1] - self.TIMES[0]

        self.SOLAR = self._get_solar_power_array(self.TIMES)

        # Reset working variables.
        self.reset()

    def reset(self):
        # Some working variables.
        self.current_step = 0
        self.speed = config["INITIAL_SPEED"]

        # Stats to keep track of.
        self.current_power_used = 0
        self.current_net_power = self.SOLAR[0]
        self.current_net_energy = self.SOLAR[0] * self.TDIFF
        self.soc = config["START_SOC"]
        self.distance_traveled_in_step = 0
        self.velocities = [0]
        self.total_distance_traveled = 0

        return self._get_observation()

    def step(self, action):
        self._take_action(action)

        # Update stats variables.
        self.velocities.append(self.speed)
        self.current_power_used = self._get_power_usage(self.speed)

        self.current_net_power = self.SOLAR[self.current_step] - self.current_power_used
        self.current_net_energy = self.current_net_energy + self.current_net_power * self.TDIFF

        self.soc = min((self.current_net_energy / config["BATTERY_CAPACITY"]) + config["START_SOC"], 1)

        self.distance_traveled_in_step = self.speed * self.TDIFF
        self.total_distance_traveled += self.distance_traveled_in_step

        # Increment current step
        self.current_step += 1

        # Calculate reward in km.
        reward = self.distance_traveled_in_step // 1_000

        # Check if done
        done = (self.current_step >= config["STEPS_IN_EPISODE"]) or (self.soc <= 0)

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, "
              f"Total dist: {self.total_distance_traveled/1000}km, "
              f"SOC: {self.soc}, "
              f"Speed: {self.speed} m/s.")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

    def _take_action(self, action):
        # Discrete(3) == [0, 1, 2]
        if action == 0: # Decelerate
            self.speed -= config["DECELERATION"]

        elif action == 2: # Accelerate.
            self.speed += config["ACCELERATION"]

    def _get_observation(self):
        return np.array([
            self.current_net_power / self.HIGHEST_SOLAR_POWER,
            self.current_net_energy / (self.HIGHEST_SOLAR_POWER * self.TDIFF),
            self.total_distance_traveled / self.MAX_DISTANCE,
            self.current_step / config["STEPS_IN_EPISODE"],
            self.soc,
            self.velocities[-1] / config["MAX_SPEED"],
        ], dtype=np.float32)

    def get_score_to_win(self):
        return 647

    """ ----------------------------------------------------------------------------------------------------
                                        Helper calculation functions
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

        aero_loss = ((velocity) ** 2 * 0.5 * 1.225 * config["FRONTAL_AREA"] * config["COEFFICIENT_OF_DRAG"]) * velocity
        rolling_loss = .8 / (.588 / 2) * 4 * velocity
        hill_loss = (9.81 * config["MASS"] * velocity * np.sin(rads))
        if (hill_loss < 0):
            hill_loss *= config["REGENERATIVE_BREAK_EFFICIENCY"]

        return 1.3 * (aero_loss + rolling_loss + hill_loss) + (config["IDLE_POWER"] * config["PACK_VOLTAGE"])