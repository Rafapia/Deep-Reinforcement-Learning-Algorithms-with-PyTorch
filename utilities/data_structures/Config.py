class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        # The random seed, for reproducibility.
        self.seed = None
        # The instantiated environment.
        self.environment = None
        # The minimum score for a game to be considered solved, if any. Defaults to None.
        self.requirements_to_solve_game = None
        # The number of episodes to run during training.
        self.num_episodes_to_run = None
        # Where to save the results of the training.
        self.file_to_save_data_results = None
        # Where to save the plot of the agent's performance.
        self.file_to_save_results_graph = None
        # How many runs per agent.
        self.runs_per_agent = None
        # Whether to visualize the overall results in a graph.
        self.visualise_overall_results = None
        # Visualize individual results.
        self.visualise_individual_results = None
        # The NN's hyperparameters.
        self.hyperparameters = None
        # Whether to use the GPU or not.
        self.use_GPU = None
        # Whether to overwrite the results files.
        self.overwrite_existing_results_file = None
        # Whether to save model after training.
        self.save_model = False
        # TODO
        self.standard_deviation_results = 1.0
        # TODO
        self.randomise_random_seed = True
        # TODO
        self.show_solution_score = False
        # Whether to log more in-depth information during training.
        self.debug_mode = False

        # An instance of the model to be used. If None, will create MLP using hyperparameters and nn_builder.
        self.model = None

        # Whether to log run on WandB.
        self.wandb_log = True
        # The tags for this run. Helps group runs.
        self.wandb_tags = "testing"
        # Who is running this run.
        self.wandb_entity = "rafael_piacsek"
        # The type of run.
        self.wandb_job_type = "train"
        # How frequently to log model.
        self.wandb_model_log_freq = 1000

