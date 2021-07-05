import sys
sys.path.insert(0, '../')

from environments.isc_environments.SimpleISC import SimpleISC
from utilities.data_structures.Config import Config
from agents.Trainer import Trainer

from agents.actor_critic_agents import A2C, A3C, DDPG, DDPG_HER

from gym.core import Wrapper
from torch.cuda import is_available

config = Config()

config.environment = Wrapper(SimpleISC(mode="DISCRETE"))
config.num_episodes_to_run = 5

config.file_to_save_data_results = "results/data_and_graphs/isc/IllinoisSolarCar_Results_Data.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/isc/IllinoisSolarCar_Results_Graph.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = False
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = is_available()
config.overwrite_existing_results_file = True
config.randomise_random_seed = False
config.save_model = False
config.seed = 0
config.debug_mode = True
config.wandb_log = True

config.hyperparameters = {
    "Actor_Critic_Agents": {

        "learning_rate": 0.0005,
        "linear_hidden_units": [128, 128, 128],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 25.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 10.0,
        "normalise_rewards": False,
        "automatically_tune_entropy_hyperparameter": True,
        "add_extra_noise": False,
        "min_steps_before_learning": 4,
        "do_evaluation_iterations": True,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.001,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "TANH",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5
        },

        "Critic": {
            "learning_rate": 0.01,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "None",
            "batch_norm": False,
            "buffer_size": 100_000,
            "tau": 0.005,
            "gradient_clipping_norm": 5
        },

        "batch_size": 3,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 10,
        "HER_sample_proportion": 0.8,
        "exploration_worker_difference": 1.0
    },
}

if __name__ == '__main__':
    AGENTS = [A2C, A3C, DDPG, DDPG_HER]

    trainer = Trainer(config, AGENTS)
    trainer.train()


