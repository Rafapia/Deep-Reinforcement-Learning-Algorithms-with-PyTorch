from environments.isc_environments.SimpleISC import SimpleISC
from utilities.data_structures.Config import Config
from agents.Trainer import Trainer

from agents.DQN_agents.DQN import DQN
from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay

from gym.core import Wrapper

config = Config()

config.seed = 1
config.environment = Wrapper(SimpleISC())
config.num_episodes_to_run = 5

config.file_to_save_data_results = "Data_and_Graphs/IllinoisSolarCar_Results_Data.pkl"
config.file_to_save_results_graph = "Data_and_Graphs/IllinoisSolarCar_Results_Graph.png"
config.show_solution_score = True
config.visualise_individual_results = True
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = True
config.randomise_random_seed = False
config.save_model = False
config.seed = 0
# config.debug_mode = True


config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.005,
        "batch_size": 128,
        "buffer_size": 100000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 150,
        "discount_rate": 0.999,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "tau": 1e-2,
        "linear_hidden_units": [256, 256, 256],
        "final_layer_activation": None,
        "y_range": (-1, 14),
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "HER_sample_proportion": 0.8,
        "learning_iterations": 1,
        "clip_rewards": False
    }
}

if __name__== '__main__':
    AGENTS = [DQN, DDQN, Dueling_DDQN, DDQN_With_Prioritised_Experience_Replay]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()


