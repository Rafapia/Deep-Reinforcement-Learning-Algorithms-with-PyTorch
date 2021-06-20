import gym
from gym import spaces
from gym.utils import seeding

from math import sin, cos, pi
import numpy as np
import datetime

from .ISC_Config import ISC_Config

class SimpleISC(gym.Env):
    metadata = {'render.mode': ['human']}

    def __init__(self, mode="DISCRETE", config=ISC_Config()):
        # Save environment configurations.
        self.config = config
        self.mode = mode

        # Gym requirement.
        if mode is "DISCRETE":
            self.action_space = spaces.Discrete(config.action_size_discrete)
        elif mode is "CONTINUOUS":
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.config.action_size_continuous,))
        else:
            raise RuntimeError(f"Invalid environment mode \"{mode}\".")

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.config.state_size,))
        self.reward_threshold = 0.0
        self.trials = 10

        # Setting all constants.
        self.speed = self.config.initial_speed

        self.HIGHEST_SOLAR_POWER = self._get_solar_power(1601294400, latitude=0, longitude=0)
        self.MAX_DISTANCE = self.config.max_speed * (self.config.end_time - self.config.start_time)
        self.reward_range = np.array([0, self.MAX_DISTANCE])

        self.TIMES = np.linspace(self.config.start_time, self.config.end_time, self.config.steps_in_episode)
        self.TDIFF = self.TIMES[1] - self.TIMES[0]

        self.SOLAR = self._get_solar_power_array(self.TIMES)

        # Reset working variables.
        self.reset()

    def reset(self):
        # Some working variables.
        self.current_step = 0
        self.speed = self.config.initial_speed

        # Stats to keep track of.
        self.current_power_used = 0
        self.current_net_power = self.SOLAR[0]
        self.current_net_energy = self.SOLAR[0] * self.TDIFF
        self.soc = self.config.initial_soc
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

        self.soc = min((self.current_net_energy / self.config.battery_capacity) + self.config.initial_soc, 1)

        self.distance_traveled_in_step = self.speed * self.TDIFF
        self.total_distance_traveled += self.distance_traveled_in_step

        # Increment current step
        self.current_step += 1

        # Calculate reward in km.
        reward = self.distance_traveled_in_step // 1_000

        # Check if done
        done = (self.current_step >= self.config.steps_in_episode) or (self.soc <= 0)

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
        if self.mode is "DISCRETE":
            if action == 0: # Decelerate
                self.speed -= self.config.deceleration

            elif action == 2: # Accelerate.
                self.speed += self.config.acceleration

        elif self.mode is "CONTINUOUS":
            self.speed += np.interp(action, [-1, 1], [-self.config.deceleration, self.config.acceleration])

    def _get_observation(self):
        return np.array([
            self.current_net_power / self.HIGHEST_SOLAR_POWER,
            self.current_net_energy / (self.HIGHEST_SOLAR_POWER * self.TDIFF),
            self.total_distance_traveled / self.MAX_DISTANCE,
            self.current_step / self.config.steps_in_episode,
            self.soc,
            self.velocities[-1] / self.config.max_speed,
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

        aero_loss = ((velocity) ** 2 * 0.5 * 1.225 * self.config.frontal_area * self.config.coefficient_of_drag) * velocity
        rolling_loss = .8 / (.588 / 2) * 4 * velocity
        hill_loss = (9.81 * self.config.mass * velocity * np.sin(rads))
        if (hill_loss < 0):
            hill_loss *= self.config.regenerative_break_efficiency

        return 1.3 * (aero_loss + rolling_loss + hill_loss) + (self.config.idle_power * self.config.pack_voltage)