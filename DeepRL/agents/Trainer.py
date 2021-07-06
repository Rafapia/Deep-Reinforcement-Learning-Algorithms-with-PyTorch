import copy
import random
import pickle
import os
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

class Trainer(object):
    """Runs games for given agents. Optionally will visualise and save the results"""
    def __init__(self, config, agent):
        # Save config and agents to run.
        self.config = config
        self.agent = agent

        # Dictionaries that hold agent groups and colors for visualization.
        self.agent_to_agent_group = self.create_agent_to_agent_group_dictionary()
        self.agent_to_color_group = self.create_agent_to_color_dictionary()

        # Dictionary where results from agents will be stored.
        self.results = None

        # Some colors.
        self.colors = ["red", "blue", "green", "orange", "yellow", "purple"]
        self.colour_ix = 0

    def create_agent_to_agent_group_dictionary(self):
        """Creates a dictionary that maps an agent to their wider agent group"""
        agent_to_agent_group_dictionary = {
            "DQN": "DQN_Agents",
            "DQN-HER": "DQN_Agents",
            "DDQN": "DQN_Agents",
            "DDQN with Prioritised Replay": "DQN_Agents",
            "DQN with Fixed Q Targets": "DQN_Agents",
            "Duelling DQN": "DQN_Agents",
            "PPO": "Policy_Gradient_Agents",
            "REINFORCE": "Policy_Gradient_Agents",
            "Genetic_Agent": "Stochastic_Policy_Search_Agents",
            "Hill Climbing": "Stochastic_Policy_Search_Agents",
            "DDPG": "Actor_Critic_Agents",
            "DDPG-HER": "Actor_Critic_Agents",
            "TD3": "Actor_Critic_Agents",
            "A2C": "Actor_Critic_Agents",
            "A3C": "Actor_Critic_Agents",
            "h-DQN": "h_DQN",
            "SNN-HRL": "SNN_HRL",
            "HIRO": "HIRO",
            "SAC": "Actor_Critic_Agents",
            "HRL": "HRL",
            "Model_HRL": "HRL",
            "DIAYN": "DIAYN",
            "Dueling DDQN": "DQN_Agents"
        }
        return agent_to_agent_group_dictionary

    def create_agent_to_color_dictionary(self):
        """Creates a dictionary that maps an agent to a hex color (for plotting purposes)
        See https://en.wikipedia.org/wiki/Web_colors and https://htmlcolorcodes.com/ for hex colors"""
        agent_to_color_dictionary = {
            "DQN": "#0000FF",
            "DQN with Fixed Q Targets": "#1F618D",
            "DDQN": "#2980B9",
            "DDQN with Prioritised Replay": "#7FB3D5",
            "Dueling DDQN": "#22DAF3",
            "PPO": "#5B2C6F",
            "DDPG": "#800000",
            "DQN-HER": "#008000",
            "DDPG-HER": "#008000",
            "TD3": "#E74C3C",
            "h-DQN": "#D35400",
            "SNN-HRL": "#800000",
            "A3C": "#E74C3C",
            "A2C": "#F1948A",
            "SAC": "#1C2833",
            "DIAYN": "#F322CD",
            "HRL": "#0E0F0F"
        }
        return agent_to_color_dictionary

    def train(self):
        """Run a set of games for the agent. Optionally visualising and/or saving the results"""
        # Create results dictionary or load in if available.
        self.results = self.create_object_to_store_results()

        # Run games and store results.
        self.run_games_for_agent(self.agent)

        # If results dta is to be saved, save it.
        if self.config.file_to_save_data_results:
            self.save_obj(self.results, self.config.file_to_save_data_results)

        return self.results

    def create_object_to_store_results(self):
        """Creates a dictionary that we will store the results in if it doesn't exist, otherwise it loads it up"""
        if self.config.overwrite_existing_results_file or \
                not self.config.file_to_save_data_results or \
                not os.path.isfile(self.config.file_to_save_data_results):
            results = {}

        else:
            results = self.load_obj(self.config.file_to_save_data_results)

        return results

    def run_games_for_agent(self, agent_class):
        """Runs a set of games for a given agent, saving the results in self.results"""
        # Stores this agent's results.
        agent_results = []
        agent_name = agent_class.agent_name
        agent_group = self.agent_to_agent_group[agent_name]
        agent_round = 1

        # For every game the agent needs to run.
        for run in range(self.config.runs_per_agent):
            # Copy configurations to be provided to agent.
            agent_config = copy.deepcopy(self.config)

            # If the env is changeable, meaning that different episodes can have different goals.
            if self.environment_has_changeable_goals(agent_config.environment) and \
                    self.agent_cant_handle_changeable_goals_without_flattening(agent_name):
                print("Flattening changeable-goal environment for agent {}".format(agent_name))
                agent_config.environment = gym.wrappers.FlattenDictWrapper(agent_config.environment,
                                                                           dict_keys=["observation", "desired_goal"])

            # Generate random seed for agent based on config.
            if self.config.randomise_random_seed:
                agent_config.seed = random.randint(0, 2**32 - 2)

            # Get specific configurations given the agent's type.
            agent_config.hyperparameters = agent_config.hyperparameters

            # Print some debug information.
            print("AGENT NAME: {}".format(agent_name))
            print("\033[1m" + "{}: {}".format(agent_round, agent_name) + "\033[0m", flush=True)

            # Instantiate agent with the given agent-type configurations.
            agent = agent_class(agent_config)

            # Get env name.
            self.environment_name = agent.environment_title

            # Print agent's hyperparameters and seed.
            print(agent.hyperparameters)
            print("RANDOM SEED ", agent_config.seed)

            # Run episodes (n is specified in config as "num_episodes_to_run")
            game_scores, rolling_scores, time_taken = agent.run_n_episodes()

            # Print run time.
            print("Time taken: {}".format(time_taken), flush=True)
            self.print_two_empty_lines()

            # Append results to this agent's result list.
            agent_results.append([game_scores, rolling_scores, len(rolling_scores), -1 * max(rolling_scores), time_taken])

            # Finally, increment agent run counter.
            agent_round += 1

        # After agent's run is over, append results to results dictionary.
        self.results[agent_name] = agent_results

    def environment_has_changeable_goals(self, env):
        """Determines whether environment is such that for each episode there is a different goal or not"""
        return isinstance(env.reset(), dict)

    def agent_cant_handle_changeable_goals_without_flattening(self, agent_name):
        """Boolean indicating whether the agent is set up to handle changeable goals"""
        return "HER" not in agent_name

    # def get_y_limits(self, results):
    #     """Extracts the minimum and maximum seen y_values from a set of results"""
    #     min_result = float("inf")
    #     max_result = float("-inf")
    #
    #     for result in results:
    #         temp_max = np.max(result)
    #         temp_min = np.min(result)
    #
    #         if temp_max > max_result:
    #             max_result = temp_max
    #
    #         if temp_min < min_result:
    #             min_result = temp_min
    #
    #     return min_result, max_result

    # def get_next_color(self):
    #     """Gets the next color in list self.colors. If it gets to the end then it starts from beginning"""
    #     self.colour_ix += 1
    #     if self.colour_ix >= len(self.colors):
    #         self.colour_ix = 0
    #
    #     color = self.colors[self.colour_ix]
    #
    #     return color

    # def ignore_points_after_game_solved(self, mean_minus_x_std, mean_results, mean_plus_x_std):
    #     """Removes the datapoints after the mean result achieves the score required to solve the game"""
    #     for ix in range(len(mean_results)):
    #         if mean_results[ix] >= self.config.environment.get_score_to_win():
    #             break
    #
    #     return mean_minus_x_std[:ix], mean_results[:ix], mean_plus_x_std[:ix]

    # def draw_horizontal_line_with_label(self, ax, y_value, x_min, x_max, label):
    #     """Draws a dotted horizontal line on the given image at the given point and with the given label"""
    #     ax.hlines(y=y_value, xmin=x_min, xmax=x_max,
    #               linewidth=2, color='k', linestyles='dotted', alpha=0.5)
    #     ax.text(x_max, y_value * 0.965, label)

    def print_two_empty_lines(self):
        print("-----------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------")
        print(" ")

    def save_obj(self, obj, name):
        """Saves given object as a pickle file"""
        if name[-4:] != ".pkl":
            name += ".pkl"

        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        """Loads a pickle file object"""
        with open(name, 'rb') as f:
            return pickle.load(f)


