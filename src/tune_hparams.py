"""
Entry Point for Hyperparameter tuning, as a meta-main for the regular training.
"""
from optuna import Study, Trial
import optuna
from main import main
import os
import yaml
from argparse import ArgumentParser
import logging

LOGGER= logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Objective:
    """
    Callable objective function which defines the optimization problem for Optuna from yaml sweep configurations.
    """
    def __init__(self, study_dir: str, base_config: str,  target_metric:str="val_b_accuracy", sweep_yaml: str=None,suggest_binary_weight=False):
        self.study_dir = study_dir
        self.base_config = base_config
        self.suggest_binary_weight = suggest_binary_weight
        self.sweep_yaml = sweep_yaml
        self.target_metric = target_metric
    
    def _legacy_hparam_definition(self, trial: Trial):
        """
        Here for transparency about how hyperparameters used to be initialized.
        """
        # lr = trial.suggest_float("learning rate", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch size", [64, 128, 256, 512, 1024])
        dropout = trial.suggest_float("dropout", 0.1, 0.9)
        weight_decay = trial.suggest_float("weight decay", 1e-5, 1e-2, log=True)
        label_smoothing = trial.suggest_float("label smoothing", 0.0, 0.3)
        if self.suggest_binary_weight:
            binary_class_weight= trial.suggest_float("binary class weight", 1, 90, log=True)


        args = [
            "--batch_size", str(batch_size),
            "--drop_rate", str(dropout),
            "--attn_drop_rate", str(dropout),
            "--drop_path_rate", str(dropout),
            "--weight_decay", str(weight_decay),
            "--smoothing", str(label_smoothing),
        ]
        if self.suggest_binary_weight:
            args.extend(["--binary_class_weight", str(binary_class_weight)])

        return args
    
    @staticmethod
    def searchspace_yaml_to_kwargs(trial: Trial, sweep_yaml: str):
        with open(sweep_yaml, 'r') as f:
            searchspace = yaml.safe_load(f)["searchspace"]
        args = []
        # Yaml format:
        # name: dtype, kwarg_names, additional_args
        # if dtype is "categorical": options: ["option1", "option2", ...]
        # if dtype is "float": low, high, log
        # if dtype is "int": low, high, log or step
        for name, hyperparam_args in searchspace.items():
            print(f"Processing hyperparameter '{name}' with args: {hyperparam_args}")
            dtype = hyperparam_args['type']
            kwarg_names = hyperparam_args.get('kwarg_names', [])
            if not isinstance(kwarg_names, list):
                kwarg_names = [kwarg_names]
            if len(kwarg_names) == 0:
                kwarg_names.append(f'--{name}') # to ensure the name is included in the args

            if dtype == "categorical":
                suggestion = trial.suggest_categorical(name, hyperparam_args['choices'])

            elif dtype == "float":
                suggestion = trial.suggest_float(
                    name, 
                    float(hyperparam_args['low']), 
                    float(hyperparam_args['high']), 
                    log=hyperparam_args.get('log', False)
                )
            elif dtype == "int":
                suggestion = trial.suggest_int(
                    name, 
                    int(hyperparam_args['low']), 
                    int(hyperparam_args['high']), 
                    log=hyperparam_args.get('log', False), 
                    step=hyperparam_args.get('step', 1)
                )
            else:
                raise ValueError(f"Unknown dtype '{dtype}' for hyperparameter '{name}' in searchspace YAML.")  

            for kwarg_name in kwarg_names:
                kwarg_name = kwarg_name.lstrip('--')  # Remove leading '--' if present
                args.append(f"--{kwarg_name}")
                args.append(str(suggestion))
        return args

    def __call__(self, trial: Trial) -> float:
        if self.sweep_yaml:
            args = self.searchspace_yaml_to_kwargs(trial, self.sweep_yaml)
        else:
            args = self._legacy_hparam_definition(trial)
        args.append("--optuna_tuning")
        args.extend(["--target_metric", self.target_metric])
        output_dir = f"{self.study_dir}/{trial.number}"
        args.extend([
            "--output_dir", output_dir,
            "--config", self.base_config,
        ])
        return main(
        argv=args,
        trial=trial,
        )

class GlobalMaxTrialCallback:
    """
    Callback for optuna to stop the study when the maximum number of trials is reached.#The code in this notebook may no longer be working, but is kept for transparency/reuse purposes.
    If this is not done, Optuna will continue to run trials until the maximum is reached within this
    single instance of the program run. Important when continuing e.g. over multible slurm jobs.
    """
    def __init__(self, max_trials: int, include_pruned=True, include_failed=False, include_running=False):
        self.max_trials = max_trials
        self.include_pruned = include_pruned
        self.include_failed = include_failed
        self.include_running = include_running

    def __call__(self, study: Study, trial: Trial):
        study_df = study.trials_dataframe()
        n_complete = (study_df["state"] == "COMPLETE").sum()
        n_pruned = (study_df["state"] == "PRUNED").sum()
        n_failed = (study_df["state"] == "FAIL").sum()
        n_running = (study_df["state"] == "RUNNING").sum()
        n_executed_trials = n_complete
        if self.include_pruned:
            n_executed_trials += n_pruned
        if self.include_failed:
            n_executed_trials += n_failed
        if self.include_running:
            n_executed_trials += n_running
        n_remaining_trials = self.max_trials - n_executed_trials
        if n_remaining_trials <= 0:
            LOGGER.info("Maximum number of trials reached. Stopping the study.")
            study.stop()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_config", type=str, default="src/config/test_training.yaml")
    parser.add_argument("--study_name", type=str, default="test_study")
    parser.add_argument("--study_dir", type=str, default="out/test_study")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--sweep_yaml", type=str, help="Path to the search space YAML file.")
    parser.add_argument("--suggest_binary_weight", action="store_true", default=False,
                        help="Suggest a binary class weight for the binary classification task.")
    parser.add_argument("--warmup_epochs", type=int, default=10)

    args = parser.parse_args()
    if not os.path.exists(args.study_dir):
        os.makedirs(args.study_dir)
    study_storage = f"sqlite:///{args.study_dir}/optuna.db"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=study_storage,
        load_if_exists=True,
        direction="maximize",
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=args.warmup_epochs,
            interval_steps=1,
        ),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True,
            group=True,
        ),
    )
    study_df = study.trials_dataframe()
    LOGGER.info(f"Loaded study with {len(study_df)} trials.")
    LOGGER.info(f"Study DataFrame:\n{study_df.head()}")

    objective = Objective(
        study_dir=args.study_dir,
        base_config=args.base_config,
        sweep_yaml=args.sweep_yaml,
        suggest_binary_weight=args.suggest_binary_weight,
        target_metric="val_balanced_accuracy",
    )
    callback = GlobalMaxTrialCallback(max_trials=args.n_trials)

    if len(study_df) > 0:    
        n_complete = (study.trials_dataframe()["state"] == "COMPLETE").sum()
        n_pruned = (study.trials_dataframe()["state"] == "PRUNED").sum()
        n_executed_trials = n_complete + n_pruned
        n_remaining_trials = args.n_trials - n_executed_trials
        LOGGER.info(f"Study already has {len(study_df)} trials. Continuing with the existing study.")
    else:
        n_remaining_trials = args.n_trials
    if n_remaining_trials <= 0:
        LOGGER.info(f'with maximum number of trials {args.n_trials} already reached. No more trials will be executed.')
    else:
        LOGGER.info(f"Starting Optuna study with {n_remaining_trials} remaining trials.")
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1,timeout=None, callbacks=[callback])
