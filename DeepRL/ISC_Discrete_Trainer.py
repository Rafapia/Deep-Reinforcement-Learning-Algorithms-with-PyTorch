from environments.isc_environments.SimpleISC import SimpleISC
from utilities.data_structures.Config import Config
from agents.Trainer import Trainer

from agents.DQN_agents import DQN, DDQN, Dueling_DDQN, DDQN_With_Prioritised_Experience_Replay, DRQN

import wandb
from gym.core import Wrapper
from torch.cuda import is_available

config = Config()

config.environment = Wrapper(SimpleISC(mode="DISCRETE"))
config.num_episodes_to_run = 1_000

config.file_to_save_data_results = "results/data_and_graphs/isc/IllinoisSolarCar_Results_Data.pkl"
config.runs_per_agent = 1
config.use_GPU = is_available()
config.overwrite_existing_results_file = True
config.randomise_random_seed = False
config.save_model = False
config.model = None
config.seed = 0

config.debug_mode = True
config.wandb_log = True
config.wandb_job_type = "testing"
config.wandb_entity = "rafael_piacsek"
config.wandb_tags = ["initial testing"]
config.wandb_model_log_freq = 1_000


config.hyperparameters = dict(
    # y_range=(-1, 14),
    HER_sample_proportion=0.8,
    alpha_prioritised_replay=0.6,
    batch_norm=False,
    batch_size=128,
    beta_prioritised_replay=0.1,
    buffer_size=100_000,
    clip_rewards=False,
    discount_rate=0.999,
    epsilon=1.0,
    epsilon_decay_rate_denominator=200,
    final_layer_activation="softmax",
    gradient_clipping_norm=5,
    incremental_td_error=1e-8,
    learning_iterations=1,
    learning_rate=0.01,
    random_episodes_to_run=config.num_episodes_to_run//5,
    tau=1e-2,
    update_every_n_steps=15,

    num_hidden_layers=None,
    hidden_layer_size=None,
    linear_hidden_units=[128, 128, 128, 128],       # Either set this, or the previous two.
)

if __name__ == '__main__':
    trainer = Trainer(config, DQN)
    trainer.train()
