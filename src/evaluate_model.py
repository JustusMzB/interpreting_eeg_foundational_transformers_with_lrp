import argparse
import json
import os
import yaml
import numpy as np
import optuna
from main import main
import logging
import pandas as pd
import glob

def hparams_to_args(params: dict, sweep_yaml: str):
    """Converts a dictionary of hyperparameters to a list of command-line arguments."""
    with open(sweep_yaml, 'r') as f:
        searchspace = yaml.safe_load(f)["searchspace"]
    
    args = []
    for name, value in params.items():
        if name not in searchspace:
            logging.warning(f"Parameter '{name}' from Optuna study not found in sweep YAML. Skipping.")
            continue
            
        hyperparam_args = searchspace[name]
        kwarg_names = hyperparam_args.get('kwarg_names', [])
        if not isinstance(kwarg_names, list):
            kwarg_names = [kwarg_names]
        if len(kwarg_names) == 0:
            # Fallback to using the parameter name itself if kwarg_names is empty
            kwarg_names.append(name)

        for kwarg_name in kwarg_names:
            kwarg_name = kwarg_name.lstrip('--')
            args.append(f"--{kwarg_name}")
            args.append(str(value))
    return args

def evaluate(args):
    # Load Optuna study
    study_storage = f"sqlite:///{args.study_dir}/optuna.db"
    study = optuna.load_study(study_name=args.study_name, storage=study_storage)
    best_params = study.best_trial.params
    
    logging.info("Best hyperparameters found:")
    logging.info(json.dumps(best_params, indent=4))

    # Convert hparams to CLI args
    hparam_args = hparams_to_args(best_params, args.sweep_yaml)

    all_run_results = []
    completed_runs = set()
    summary_table_path = os.path.join(args.output_dir, "evaluation_summary.csv")
    if args.kfold is not None:
        assert args.kfold > 1, "k-fold cross-validation requires at least 2 folds."
        if args.n_runs is not None:
            assert args.n_runs == args.kfold, "Number of runs must match number of folds for k-fold cross-validation."
        else:
            n_folds = args.kfold

    # Load existing results if available to allow for continuation
    if os.path.exists(summary_table_path):
        logging.info(f"Found existing summary table at {summary_table_path}. Loading completed runs.")
        results_df = pd.read_csv(summary_table_path)
        all_run_results = results_df.to_dict('records')
        completed_runs = results_df['run'].tolist()
        logging.info(f"Found {len(completed_runs)} completed runs: {sorted(completed_runs)}")

    for i in range(args.n_runs):
        run_output_dir = os.path.join(args.output_dir, f"run_{i}")
        # Check for attribution files
        attribution_path_pattern = os.path.join(run_output_dir, '*', 'version_*', 'attribution_results_*.parquet')
        attribution_files_exist = glob.glob(attribution_path_pattern)

        if i in completed_runs and attribution_files_exist:
            logging.info(f"--- Run {i+1}/{args.n_runs} already complete with attributions. Skipping. ---")
            continue
        elif i in completed_runs and not attribution_files_exist:
            logging.warning(f"Run {i+1}/{args.n_runs} is in summary but missing attribution files. Re-running.")

        if args.seed is not None:
            # If a base seed is provided, create a deterministic seed for each run
            run_seed = args.seed + i
        else:
            # If no seed is provided, generate a random one for each run
            run_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
            
        logging.info(f"\n--- Starting Run {i+1}/{args.n_runs} with seed {run_seed} ---")
        
        output_dir = os.path.join(args.output_dir, f"run_{i}")
        
        run_args = [
            "--config", args.base_config,
            "--output_dir", output_dir,
            "--seed", str(run_seed),
            "--target_metric", args.target_metric,
            "--do_testing",
            "--attribute",
        ]
        if args.subsample_attributions is not None:
            run_args.append(f"--attributions_per_example={args.subsample_attributions}")
        if args.debug:
            run_args.append("--debug")
        if args.num_worker_override is not None:
            run_args.append(f"--num_workers={args.num_worker_override}")
        if args.kfold is not None:
            run_args.append(f"--kfold={args.kfold}")
            run_args.append(f"--fold_index={i}")
            # run_args.append(f"--kfold_seed={args.kfold_seed}")
        run_args.extend(hparam_args)
        
        main(argv=run_args)
        
        # Find the results file for this run
        results_path_pattern = os.path.join(output_dir, '*', 'version_*', 'test_results.json')
        found_files = glob.glob(results_path_pattern)
        if found_files:
            with open(found_files[0], 'r') as f:
                run_results = json.load(f)
                run_results['run'] = i
                run_results['seed'] = run_seed
                all_run_results.append(run_results)
            
            # --- Incremental Save ---
            # Create and save summary table after each successful run
            results_df = pd.DataFrame(all_run_results)
            results_df.to_csv(summary_table_path, index=False)
            logging.info(f"Updated summary table saved to {summary_table_path}")

            # Calculate summary statistics and save to a final JSON file
            summary_stats = results_df.describe().to_dict()
            final_results = {
                "best_hparams": best_params,
                "evaluation_runs": all_run_results,
                "summary_statistics": summary_stats
            }
            
            with open(args.results_file, 'w') as f:
                json.dump(final_results, f, indent=4)
            logging.info(f"Updated final results saved to {args.results_file}")

        else:
            logging.warning(f"Could not find test_results.json for run {i} in {output_dir}")

    # Final summary logging
    if all_run_results:
        logging.info("\n--- Evaluation Complete ---")
        logging.info(f"Final results saved to {args.results_file}")
        logging.info("Summary Statistics:")
        with open(args.results_file, 'r') as f:
            final_results = json.load(f)
            logging.info(json.dumps(final_results["summary_statistics"], indent=4))
    else:
        logging.warning("No results were collected from any of the runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the best hyperparameters from an Optuna study."
    )
    parser.add_argument("--study_dir", type=str, required=True, help="Directory of the Optuna study (containing optuna.db).")
    parser.add_argument("--study_name", type=str, required=True, help="Name of the Optuna study.")
    parser.add_argument("--base_config", type=str, required=True, help="Path to the base YAML config file.")
    parser.add_argument("--sweep_yaml", type=str, required=True, help="Path to the search space YAML file used for the study.")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of evaluation runs with different seeds. If k_fold is set, has to be number of folds.")
    parser.add_argument("--kfold", type=int, default=None, help="Number of folds for k-fold cross-validation. If set, overrides the k-fold setting in the config.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed for reproducibility. If not set, a random seed will be used for each run.")
    parser.add_argument("--output_dir", type=str, default="out/evaluation", help="Base directory to save outputs for each run.")
    parser.add_argument("--results_file", type=str, default="results.json", help="File to save the final evaluation results.")
    parser.add_argument("--target_metric", type=str, default="val_balanced_accuracy", help="Target metric to report.")
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode for more verbose output.")
    parser.add_argument("--num_worker_override", type=int, default=None, help="Override the number of workers in the config file. If not set, uses the value from the config.")
    parser.add_argument("--subsample_attributions", type=int, default=None, help="Subsample the attribution results to this many samples (keeping balance of binary classes.). If not set, uses all samples.")
    cli_args = parser.parse_args()
    logging.basicConfig(level=cli_args.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    os.makedirs(cli_args.output_dir, exist_ok=True)
    # If results_file is just a filename, prepend the output_dir
    if os.path.dirname(cli_args.results_file) == '':
        cli_args.results_file = os.path.join(cli_args.output_dir, cli_args.results_file)
    evaluate(cli_args)
