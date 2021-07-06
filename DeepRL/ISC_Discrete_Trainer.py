from environments.isc_environments.SimpleISC import SimpleISC
from utilities.data_structures.Config import Config
from agents.Trainer import Trainer

from agents.DQN_agents import DQN, DDQN, Dueling_DDQN, DDQN_With_Prioritised_Experience_Replay, DRQN

import wandb
from gym.core import Wrapper
from torch.cuda import is_available

config = Config()

config.environment = Wrapper(SimpleISC(mode="DISCRETE"))
config.num_episodes_to_run = 2_000

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
config.wandb_tags = ["initial testing", "initial sweep"]
config.wandb_model_log_freq = 1_000


config.hyperparameters = dict(
    learning_rate=0.01,
    batch_size=128,
    buffer_size=1_000_000,
    epsilon=1.0,
    epsilon_decay_rate_denominator=150,
    discount_rate=0.999,
    alpha_prioritised_replay=0.6,
    beta_prioritised_replay=0.1,
    incremental_td_error=1e-8,
    update_every_n_steps=15,
    tau=1e-2,
    num_hidden_layers=2,
    hidden_layer_size=128,
    linear_hidden_units=None,       # Either set this, or the previous two.
    final_layer_activation="softmax",
    # y_range=(-1, 14),
    batch_norm=False,
    gradient_clipping_norm=5,
    HER_sample_proportion=0.8,
    learning_iterations=1,
    clip_rewards=False
)

if __name__ == '__main__':
    trainer = Trainer(config, DQN)
    trainer.train()
