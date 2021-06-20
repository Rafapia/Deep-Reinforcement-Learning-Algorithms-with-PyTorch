from dateutil import tz
import datetime

class ISC_Config(object):
    def __init__(self):
        # Initial configurations for the env.
        self.steps_in_episode = 250
        self.max_speed = 35 # In m/s.

        self.initial_speed = 0 # In m/s.
        self.initial_soc = 1.0
        self.start_time = datetime.datetime.timestamp(datetime.datetime(2021, 7, 21, 8, 0, tzinfo=tz.tzoffset('CST', -5*3600))) # 8:00 AM someday in july
        self.end_time = datetime.datetime.timestamp(datetime.datetime(2021, 7, 21, 17, 0, tzinfo=tz.tzoffset('CST', -5*3600))) # 5:00 PM that day

        # Car properties and constants.
        self.pack_voltage = 100
        self.battery_capacity = 5200 * 3600 # In Ws (Watt-second).
        self.regenerative_break_efficiency = 0.1
        self.mass = 250
        self.frontal_area = 0.78
        self.coefficient_of_drag = 0.116
        self.idle_power = 0.39
        self.acceleration = 2.5
        self.deceleration = 1.5

        # Environment settings.
        self.state_size = 6

        # Discrete
        self.action_size_discrete = 3
        # Continuous
        self.action_size_continuous = 1
