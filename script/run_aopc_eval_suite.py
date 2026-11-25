#!/usr/bin/env python
"""
Runner script for Arousal and PhysioMI AOPC (Area Over Perturbation Curve) evaluations.
Traverses experiment run directories, computes/loads attributions and checkpoints,
runs AOPC faithfulness evaluation, and aggregates summary statistics.
"""
import argparse
import glob
import json
import logging
import shutil
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import yaml
import warnings
import os
os.environ["PYTHONWARNINGS"] = ("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pl_bolts")

# Add src to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluate.aopc import (evaluate_aopc_for_dataset, random_order, get_best_device,
                           aopc_per_sample, aopc_curve, legacy_running_mean_score,
                           check_attribution_alignment, per_sample_order, global_order,
                           PER_SAMPLE_ORDERINGS, GLOBAL_ORDERINGS)
from evaluate.analysis_utils import AttributionAnalyzer, ExperimentResults
from evaluate.aopc_table import write_aopc_table
from main import build_data_module
from models.lighting_labram import NeuralTransformerLightning

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger("aopc_suite_runner")

# Baseline / main experiment model evaluation directory names (relative to base_dir or absolute)
AROUSAL_MODEL_EXPERIMENTS = {
    "arousal_pretrained": "true_binary_arousal_pretrained_evaluation",
    "arousal_from_scratch": "true_binary_arousal_from_scratch_evaluation",
    "arousal_frozen": "arousal_frozen_evaluation",
}

PHYSIOMI_MODEL_EXPERIMENTS = {
    "physiomi_pretrained": "true_binary_physionet_pretrained_evaluation",
    "physiomi_from_scratch": "true_binary_physionet_from_scratch_evaluation",
    "physiomi_frozen": "physionet_binary_frozen_evaluation",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Run batch AOPC evaluation across experiment suites.")
    parser.add_argument(
        "--suite",
        choices=["arousal", "physiomi", "all"],
        default="all",
        help="Experiment suite to evaluate.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.path.expanduser("~/logs") if not os.path.exists("/ptmp") else f"/ptmp/{os.environ.get('USER', 'user')}/logs",
        help="Base directory containing the model evaluation log directories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save aggregated AOPC suite summary results. Defaults to base_dir/aopc_suite_results.",
    )
    parser.add_argument(
        "--run_output_dir",
        type=str,
        default=None,
        help="Alternate directory to save per-run AOPC results (to avoid cluttering exp_dir).",
    )
    parser.add_argument(
        "--aopc_perturbation_strategy", "--aopc_baseline",
        choices=["global_mean", "uniform_noise"],
        default="uniform_noise",
        dest="aopc_perturbation_strategy",
        help="Strategy for spatial channel perturbation in AOPC.",
    )
    parser.add_argument(
        "--aopc_split",
        choices=["test", "val"],
        default="test",
        help="Data split to evaluate for AOPC.",
    )
    parser.add_argument(
        "--global_ordering",
        choices=list(GLOBAL_ORDERINGS),
        default="magnitude",
        help="Global channel order. 'magnitude' = |visualised global heatmap|; see evaluate.aopc.global_order.",
    )
    parser.add_argument(
        "--per_sample_ordering",
        choices=list(PER_SAMPLE_ORDERINGS),
        default="prediction",
        help="Per-sample channel order; see evaluate.aopc.per_sample_order.",
    )
    parser.add_argument(
        "--aopc_max_steps",
        type=int,
        default=None,
        help="Max number of spatial perturbation steps for AOPC.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--mini_n_samples",
        type=int,
        default=None,
        help="Subsample number of test examples per run for quick evaluation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for AOPC perturbation forward pass evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for sample selection and the random-order baseline.",
    )
    parser.add_argument(
        "--overwrite_and_backup", "--backup_existing",
        action="store_true",
        dest="overwrite_and_backup",
        help="Move existing AOPC evaluation reports/results to a timestamped backup directory and start a fresh evaluation instead of skipping them.",
    )
    return parser


def find_run_directories(exp_dir):
    """
    Finds all run directories (e.g. run_0, run_1, ...) inside an experiment folder,
    or returns [exp_dir] if exp_dir itself is a single run folder.
    """
    if not os.path.exists(exp_dir):
        LOGGER.warning(f"Experiment directory does not exist: {exp_dir}")
        return []

    sub_runs = sorted(glob.glob(os.path.join(exp_dir, "run_*")))
    valid_sub_runs = [r for r in sub_runs if os.path.isdir(r)]

    if valid_sub_runs:
        return valid_sub_runs

    ckpts = glob.glob(os.path.join(exp_dir, "**", "*.ckpt"), recursive=True)
    if ckpts:
        return [exp_dir]

    return []


from main import build_data_module, parse_args

PRODUCTION_CONFIG_MAP = {
    "arousal_pretrained": "arousal_finetuned.yaml",
    "arousal_from_scratch": "arousal_from_scratch.yaml",
    "arousal_frozen": "arousal_frozen.yaml",
    "physiomi_pretrained": "physionet_finetuned.yaml",
    "physiomi_from_scratch": "physionet_from_scratch.yaml",
    "physiomi_frozen": "physionet_frozen.yaml",
}

def extract_args_from_run(run_dir, exp_name):
    """
    Constructs the exact dataloader arguments by parsing the original production YAML config.
    """
    if exp_name in PRODUCTION_CONFIG_MAP:
        config_file = PRODUCTION_CONFIG_MAP[exp_name]
    elif "arousal" in exp_name:
        config_file = "arousal_finetuned.yaml"
    elif "physiomi" in exp_name or "physionet" in exp_name:
        config_file = "physionet_finetuned.yaml"
    else:
        raise ValueError(f"Could not map experiment '{exp_name}' to a production config.")
        
    config_path = os.path.join(SRC_DIR, "config", "production", config_file)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Production config not found: {config_path}")
        
    args = parse_args(["--config", config_path])
    
    # We still need to ensure datamodule uses a valid data_dir if it was left blank/relative
    if hasattr(args, 'dataset_path') and args.dataset_path:
        if not os.path.isabs(args.dataset_path):
            args.dataset_path = os.path.join(PROJECT_ROOT, args.dataset_path)
            
    if hasattr(args, 'data_dir') and args.data_dir:
        if not os.path.isabs(args.data_dir):
            args.data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    
    return args


def evaluate_experiment_folder(exp_name, exp_dir, args):
    """
    Evaluates all runs inside an experiment directory for AOPC metrics.
    """
    LOGGER.info("=" * 80)
    LOGGER.info(f"STARTING AOPC EVALUATION FOR EXPERIMENT: {exp_name}")
    LOGGER.info(f"Directory: {exp_dir}")
    LOGGER.info("=" * 80)

    suffix = f"{args.aopc_perturbation_strategy}_ps-{args.per_sample_ordering}_gl-{args.global_ordering}"
    if getattr(args, 'mini_n_samples', None):
        suffix += f"_mini{args.mini_n_samples}"
        
    base_out_dir = os.path.join(args.run_output_dir, exp_name) if getattr(args, 'run_output_dir', None) else exp_dir
    out_dir = os.path.join(base_out_dir, f"aopc_results_{suffix}")
    out_path = os.path.join(out_dir, "aopc_evaluation_results.json")
    
    results_list = []
    if getattr(args, "overwrite_and_backup", False):
        if os.path.exists(out_dir) and os.listdir(out_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{out_dir}_backup_{timestamp}"
            try:
                shutil.move(out_dir, backup_dir)
                LOGGER.info(f"Moved existing AOPC results from '{out_dir}' to backup directory '{backup_dir}'")
            except Exception as e:
                LOGGER.warning(f"Could not move existing results from '{out_dir}' to backup: {e}")
            os.makedirs(out_dir, exist_ok=True)
    elif os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                results_list = json.load(f)
            LOGGER.info(f"Loaded {len(results_list)} previously evaluated runs from {out_path}")
        except Exception as e:
            LOGGER.warning(f"Could not load previous results from {out_path}: {e}")

    run_dirs = find_run_directories(exp_dir)
    if not run_dirs:
        LOGGER.warning(f"No valid run directories found in {exp_dir}. Skipping.")
        return []

    evaluated_runs = {r.get("run_dir") for r in results_list}

    analyzer = AttributionAnalyzer(ExperimentResults(exp_dir))
    device = get_best_device()

    for run_idx, r_dir in enumerate(run_dirs):
        if r_dir in evaluated_runs:
            LOGGER.info(f"Skipping already evaluated run: {r_dir}")
            continue
            
        LOGGER.info(f"\n--- Evaluating AOPC for run: {r_dir} ---")
        try:
            # 1. Load Model
            ckpt_paths = glob.glob(os.path.join(r_dir, "**", "*target_best.ckpt"), recursive=True)
            if not ckpt_paths:
                ckpt_paths = glob.glob(os.path.join(r_dir, "**", "last.ckpt"), recursive=True)
            if not ckpt_paths:
                raise FileNotFoundError(f"No checkpoints found in {r_dir}")
                
            ckpt_path = ckpt_paths[0]
            model = NeuralTransformerLightning.load_from_checkpoint(ckpt_path, strict=False, weights_only=False)
            model.eval()

            # 2. Build Datamodule
            run_args = extract_args_from_run(r_dir, exp_name)
            run_args.batch_size = args.batch_size
            run_args.num_workers = args.workers
            # Fix potentially relative path
            if hasattr(run_args, 'dataset_path') and not os.path.isabs(run_args.dataset_path):
                run_args.dataset_path = os.path.join(SRC_DIR, run_args.dataset_path)

            datamodule = build_data_module(run_args)
            if args.aopc_split == "test":
                dataloader = datamodule.test_dataloader()
            else:
                dataloader = datamodule.val_dataloader()
                
            # 3. Load Attributions
            LOGGER.info("Loading attributions...")
            attr_df = analyzer.load_run_data_to_memory(run_idx, split=args.aopc_split)
            
            # Robust alignment: Ensure exactly 1 row per example_index (handling both single-logit and multi-logit files)
            if 'example_index' in attr_df.columns:
                if attr_df['example_index'].duplicated().any():
                    if 'attributed_class_logit' in attr_df.columns and 'prediction' in attr_df.columns:
                        matched_df = attr_df[attr_df['attributed_class_logit'] == attr_df['prediction']]
                        missing_indices = set(attr_df['example_index']) - set(matched_df['example_index'])
                        if missing_indices:
                            fallback_df = attr_df[attr_df['example_index'].isin(missing_indices)].drop_duplicates(subset=['example_index'])
                            attr_df = pd.concat([matched_df, fallback_df], ignore_index=True)
                        else:
                            attr_df = matched_df
                    else:
                        attr_df = attr_df.drop_duplicates(subset=['example_index'])
                
                attr_df = attr_df.sort_values('example_index').reset_index(drop=True)
                
            # Refuse to run if the attributions refer to a different sample order than the dataset.
            check_attribution_alignment(
                attr_df["example_index"].values, attr_df["label"].values, dataloader.dataset,
                sample_keys=attr_df["sample_key"].values if "sample_key" in attr_df.columns else None,
            )
            LOGGER.info("Attribution/dataset alignment check passed.")

            attributions = np.stack(attr_df["attribution"].values)
            attr_tensor = torch.tensor(attributions, dtype=torch.float32)
            
            eval_limit = min(args.mini_n_samples, len(attr_tensor)) if args.mini_n_samples else len(attr_tensor)
            
            # Randomly sample to avoid ordering bias, keeping attr and data aligned
            import torch.utils.data as data_utils
            gen = torch.Generator().manual_seed(args.seed)
            all_indices = torch.randperm(len(attr_tensor), generator=gen).tolist()
            selected_indices = all_indices[:eval_limit]
            
            attr_tensor = attr_tensor[selected_indices]
            subset_dataset = data_utils.Subset(dataloader.dataset, selected_indices)
            sampled_dataloader = data_utils.DataLoader(
                subset_dataset,
                batch_size=dataloader.batch_size or args.batch_size,
                collate_fn=dataloader.collate_fn,
                num_workers=dataloader.num_workers
            )

            def get_dl_iter():
                for b in sampled_dataloader:
                    yield b

            import math
            total_batches = math.ceil(eval_limit / (dataloader.batch_size or args.batch_size))

            num_samples, num_channels, _ = attr_tensor.shape

            LOGGER.info("Computing Grad x Input attributions...")
            model.to(device)
            model.eval()
            grad_x_inputs = []
            model_logits = []
            
            import inspect
            sig = inspect.signature(model.forward)
            valid_kwargs = set(sig.parameters.keys())
            
            for batch in sampled_dataloader:
                if isinstance(batch, dict):
                    b_samples = batch.get('signal', batch.get('samples')).to(device)
                    b_target = batch.get('label', batch.get('target'))
                    if b_target is not None:
                        b_target = b_target.to(device)
                    model_kwargs = {}
                    for k, v in batch.items():
                        if k not in ['signal', 'samples', 'label', 'target']:
                            mapped_k = 'input_chans' if k == 'eeg_channel_indices' else k
                            if mapped_k in valid_kwargs or 'kwargs' in valid_kwargs:
                                model_kwargs[mapped_k] = v.to(device) if isinstance(v, torch.Tensor) else v
                else:
                    b_samples = batch[0].to(device)
                    b_target = batch[1].to(device) if len(batch) > 1 else None
                    model_kwargs = {}
                
                b_samples.requires_grad_(True)
                if b_samples.grad is not None:
                    b_samples.grad.zero_()
                model.zero_grad()
                
                outputs = model(b_samples, **model_kwargs)
                
                is_binary = outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1)
                if is_binary:
                    grad_target = outputs.view(-1)
                else:
                    if b_target is None:
                        b_target = outputs.argmax(dim=-1)
                    grad_target = outputs[torch.arange(outputs.size(0)), b_target]
                
                model_logits.append(outputs.detach().reshape(outputs.shape[0], -1)[:, 0].cpu())
                grad_target.sum().backward()
                
                gxi = b_samples.grad * b_samples # (batch, channels, time)
                grad_x_inputs.append(gxi.detach().cpu())
                
            grad_x_inputs = torch.cat(grad_x_inputs, dim=0) # (num_samples, channels, time)
            if grad_x_inputs.dim() > 2:
                gxi_agg = grad_x_inputs.sum(dim=list(range(2, grad_x_inputs.dim())))
            else:
                gxi_agg = grad_x_inputs
            
            # G×I is unbounded and its scale varies strongly between samples. Normalising each sample by
            # its max |channel score| keeps a few large samples from dominating the global G×I aggregate.
            # Per-sample orders are unaffected (positive rescaling). LRP is deliberately NOT normalised,
            # so its global aggregate stays identical to the heatmaps shown in the paper's figures.
            max_abs = gxi_agg.abs().amax(dim=1, keepdim=True) + 1e-8
            gxi_norm = gxi_agg / max_abs
            
            model_logits = torch.cat(model_logits)  # (num_samples,), same sample order as attr_tensor

            # 4. Perturbation orders. LRP and G×I go through identical ordering functions.
            LOGGER.info(f"Orders: per-sample '{args.per_sample_ordering}', global '{args.global_ordering}' "
                        f"(global aggregate over {num_samples} samples)")
            morf_order_tensor = per_sample_order(attr_tensor, model_logits, args.per_sample_ordering)
            global_order_tensor = global_order(attr_tensor, model_logits, args.global_ordering)
            gxi_morf_order_tensor = per_sample_order(gxi_norm, model_logits, args.per_sample_ordering)
            gxi_global_order_tensor = global_order(gxi_norm, model_logits, args.global_ordering)

            LOGGER.info("Computing per-sample Random order...")
            rand_order_tensor = random_order(num_samples, num_channels, device='cpu', generator=gen)
            
            if args.aopc_max_steps:
                LOGGER.info(f"Truncating evaluation to first {args.aopc_max_steps} steps...")
                global_order_tensor = global_order_tensor[:args.aopc_max_steps]
                morf_order_tensor = morf_order_tensor[:, :args.aopc_max_steps]
                rand_order_tensor = rand_order_tensor[:, :args.aopc_max_steps]
                gxi_morf_order_tensor = gxi_morf_order_tensor[:, :args.aopc_max_steps]
                gxi_global_order_tensor = gxi_global_order_tensor[:args.aopc_max_steps]

            # 5. Run Evaluations
            morf_cumsum, morf_logits = evaluate_aopc_for_dataset(
                model=model, dataloader=get_dl_iter(), order=morf_order_tensor,
                perturbation_strategy=args.aopc_perturbation_strategy, device=device,
                total_batches=total_batches, use_targets_from_dataloader=False
            )
            rand_cumsum, rand_logits = evaluate_aopc_for_dataset(
                model=model, dataloader=get_dl_iter(), order=rand_order_tensor,
                perturbation_strategy=args.aopc_perturbation_strategy, device=device,
                total_batches=total_batches, use_targets_from_dataloader=False
            )
            global_cumsum, global_logits = evaluate_aopc_for_dataset(
                model=model, dataloader=get_dl_iter(), order=global_order_tensor,
                perturbation_strategy=args.aopc_perturbation_strategy, device=device,
                total_batches=total_batches, use_targets_from_dataloader=False
            )
            gxi_morf_cumsum, gxi_morf_logits = evaluate_aopc_for_dataset(
                model=model, dataloader=get_dl_iter(), order=gxi_morf_order_tensor,
                perturbation_strategy=args.aopc_perturbation_strategy, device=device,
                total_batches=total_batches, use_targets_from_dataloader=False
            )
            gxi_global_cumsum, gxi_global_logits = evaluate_aopc_for_dataset(
                model=model, dataloader=get_dl_iter(), order=gxi_global_order_tensor,
                perturbation_strategy=args.aopc_perturbation_strategy, device=device,
                total_batches=total_batches, use_targets_from_dataloader=False
            )

            # 6. Calculate summary scalar metrics and full curves
            legacy_scores = {}
            per_sample_aopc = {}

            def calc_score_and_curve(cumsum, name=None):
                # Eq. 2 of the paper: per-sample sum of drops / (L + 1), averaged over samples.
                per_sample = aopc_per_sample(cumsum)
                if name is not None:
                    legacy_scores[name] = legacy_running_mean_score(cumsum)
                    per_sample_aopc[name] = per_sample.cpu().tolist()
                return per_sample.mean().item(), aopc_curve(cumsum).cpu().tolist()
            
            morf_score, morf_curve = calc_score_and_curve(morf_cumsum, 'morf')
            random_score, rand_curve = calc_score_and_curve(rand_cumsum, 'rand')
            summary_score, global_curve = calc_score_and_curve(global_cumsum, 'global')
            gxi_morf_score, gxi_morf_curve = calc_score_and_curve(gxi_morf_cumsum, 'gxi_morf')
            gxi_global_score, gxi_global_curve = calc_score_and_curve(gxi_global_cumsum, 'gxi_global')
            
            # 7. Extract example logits
            def get_example_logits(logits_tensor, num_examples=4):
                if logits_tensor is None:
                    return []
                n_available = min(num_examples, logits_tensor.shape[1])
                # logits_tensor: (order_steps+1, batch_size, ...)
                # we want a list of num_examples trajectories
                examples = []
                for i in range(n_available):
                    traj = logits_tensor[:, i].cpu().tolist()
                    examples.append(traj)
                return examples
                
            morf_logit_examples = get_example_logits(morf_logits)
            rand_logit_examples = get_example_logits(rand_logits)
            global_logit_examples = get_example_logits(global_logits)
            gxi_morf_logit_examples = get_example_logits(gxi_morf_logits)
            gxi_global_logit_examples = get_example_logits(gxi_global_logits)

            run_name = os.path.basename(r_dir)
            results_list.append({
                "experiment": exp_name,
                "run_dir": r_dir,
                "run": run_name,
                "split": args.aopc_split,
                "perturbation_strategy": args.aopc_perturbation_strategy,
                "baseline": args.aopc_perturbation_strategy,
                "aopc_score": summary_score,
                "morf_score": morf_score,
                "random_score": random_score,
                "morf_curve": morf_curve,
                "rand_curve": rand_curve,
                "global_curve": global_curve,
                "gxi_morf_score": gxi_morf_score,
                "gxi_global_score": gxi_global_score,
                "gxi_morf_curve": gxi_morf_curve,
                "gxi_global_curve": gxi_global_curve,
                "morf_logit_examples": morf_logit_examples,
                "rand_logit_examples": rand_logit_examples,
                "global_logit_examples": global_logit_examples,
                "gxi_morf_logit_examples": gxi_morf_logit_examples,
                "gxi_global_logit_examples": gxi_global_logit_examples,
                "summary_display_aopc_score": summary_score,
                "morf_aopc_score": morf_score,
                "random_aopc_score": random_score,
                "checkpoint_path": ckpt_path,
                "attribution_path": analyzer.results.get_attribution_path(run_idx, split=args.aopc_split),
                "score_definition": "Eq. 2: sum of drops / (L+1), sign from model prediction",
                "legacy_running_mean_scores": legacy_scores,
                "per_sample_aopc": per_sample_aopc,
                "seed": args.seed,
                "per_sample_ordering": args.per_sample_ordering,
                "global_ordering": args.global_ordering,
                "predicted_class": (model_logits > 0).long().tolist(),
                "aopc_by_predicted_class": {
                    name: {str(c): float(np.mean([v for v, p in zip(vals, (model_logits > 0).long().tolist()) if p == c]))
                           for c in (0, 1) if any(p == c for p in (model_logits > 0).long().tolist())}
                    for name, vals in per_sample_aopc.items()
                }
            })
            LOGGER.info(f"Successfully evaluated {run_name}. Global AOPC: {summary_score:.4f} (MoRF: {morf_score:.4f}, Random: {random_score:.4f})")
        except Exception as e:
            LOGGER.error(f"Failed AOPC evaluation for {r_dir}: {e}", exc_info=True)
            
        finally:
            import gc
            # Aggressive cleanup to avoid dataloader workers accumulating open files
            if 'model' in locals(): del model
            if 'datamodule' in locals(): del datamodule
            if 'dataloader' in locals(): del dataloader
            if 'sampled_dataloader' in locals(): del sampled_dataloader
            if 'attr_df' in locals(): del attr_df
            if 'attr_tensor' in locals(): del attr_tensor
            gc.collect()

    if results_list:
        # Save results
        suffix = f"{args.aopc_perturbation_strategy}_ps-{args.per_sample_ordering}_gl-{args.global_ordering}"
        if getattr(args, 'mini_n_samples', None):
            suffix += f"_mini{args.mini_n_samples}"
            
        base_out_dir = os.path.join(args.run_output_dir, exp_name) if getattr(args, 'run_output_dir', None) else exp_dir
        out_dir = os.path.join(base_out_dir, f"aopc_results_{suffix}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "aopc_evaluation_results.json")
        with open(out_path, "w") as f:
            json.dump(results_list, f, indent=4)
        LOGGER.info(f"Saved experiment AOPC results to {out_path}")

        # Table A5 per run: CSV + booktabs LaTeX, including the ratio to the random baseline.
        for res in results_list:
            if "per_sample_aopc" not in res:
                continue  # results from before the Eq. 2 fix carry no per-sample values
            table_prefix = os.path.join(out_dir, f"aopc_table_{res['run']}")
            try:
                write_aopc_table(res, table_prefix)
                LOGGER.info(f"Saved AOPC table to {table_prefix}.csv / .tex")
            except Exception as e:
                LOGGER.warning(f"Could not write AOPC table for {res['run']}: {e}")
        
        # Write text report
        methodology_path = os.path.join(out_dir, "aopc_evaluation_methodology.txt")
        try:
            with open(methodology_path, "w") as f:
                f.write("AOPC Evaluation Methodology\n")
                f.write("===========================\n\n")
                f.write(f"Perturbation strategy: {args.aopc_perturbation_strategy} (replacement value computed per sample and channel)\n")
                f.write(f"Per-sample ordering: {args.per_sample_ordering}\n")
                f.write(f"Global ordering: {args.global_ordering}\n")
                f.write(f"Max perturbation steps L: {args.aopc_max_steps or 'all channels'}\n")
                f.write(f"Seed: {args.seed}\n\n")
                f.write("Score (paper Eq. 2): AOPC = 1/(L+1) * sum_{s=1..L} sign(f(x)) * (f(x) - f(x_s)), per sample, "
                        "averaged over samples. sign(f(x)) is the sign of the model's own (single) logit, "
                        "not the dataset label. Perturbations are cumulative along the channel order.\n")
                f.write("Orders: LRP and G×I use identical ordering functions (evaluate.aopc.per_sample_order / "
                        "global_order) on channel scores summed over time. G×I channel scores are normalised per "
                        "sample by their max |value| (G×I is unbounded); LRP scores are used as visualised.\n")
                f.write("Random baseline: seeded random permutation per sample.\n")
                f.write("Per-sample values and a breakdown by predicted class are stored in the results JSON "
                        "('per_sample_aopc', 'aopc_by_predicted_class'). 'legacy_running_mean_scores' holds the "
                        "pre-fix score mean_k(cumsum_k/k) for comparison only.\n")
            LOGGER.info(f"Saved methodology report to {methodology_path}")
        except Exception as e:
            LOGGER.warning(f"Could not save methodology report to {methodology_path}: {e}")

        # Save dataframe representation
        summary_df = pd.DataFrame(results_list)
        csv_out = os.path.join(out_dir, "aopc_experiment_summary.csv")
        summary_df.to_csv(csv_out, index=False)
        LOGGER.info(f"Saved experiment AOPC summary to {csv_out}")

    return results_list


def main():
    parser = build_parser()
    args = parser.parse_args()

    suffix = f"{args.aopc_perturbation_strategy}_ps-{args.per_sample_ordering}_gl-{args.global_ordering}"
    if args.mini_n_samples:
        suffix += f"_mini{args.mini_n_samples}"

    output_dir = args.output_dir or os.path.join(args.base_dir, f"aopc_suite_results_{suffix}")
    os.makedirs(output_dir, exist_ok=True)

    all_experiments = {}

    if args.suite in ["arousal", "all"]:
        for name, path_rel in AROUSAL_MODEL_EXPERIMENTS.items():
            all_experiments[name] = os.path.join(args.base_dir, path_rel)

    if args.suite in ["physiomi", "all"]:
        for name, path_rel in PHYSIOMI_MODEL_EXPERIMENTS.items():
            all_experiments[name] = os.path.join(args.base_dir, path_rel)

    LOGGER.info(f"Starting AOPC evaluation suite on {len(all_experiments)} experiment configurations.")
    LOGGER.info(f"Base logs directory: {args.base_dir}")
    LOGGER.info(f"Output directory: {output_dir}")

    if getattr(args, "overwrite_and_backup", False) and os.path.exists(output_dir):
        master_csv = os.path.join(output_dir, "aopc_suite_master_summary.csv")
        master_json = os.path.join(output_dir, "aopc_suite_master_summary.json")
        if os.path.exists(master_csv) or os.path.exists(master_json):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = os.path.join(output_dir, f"backup_{timestamp}")
            os.makedirs(backup_subdir, exist_ok=True)
            for f_name in ["aopc_suite_master_summary.csv", "aopc_suite_master_summary.json"]:
                f_path = os.path.join(output_dir, f_name)
                if os.path.exists(f_path):
                    shutil.move(f_path, os.path.join(backup_subdir, f_name))
            LOGGER.info(f"Moved existing master suite summaries in '{output_dir}' to '{backup_subdir}'")

    all_suite_results = []
    for exp_name, exp_dir in all_experiments.items():
        res_list = evaluate_experiment_folder(exp_name, exp_dir, args)
        all_suite_results.extend(res_list)

    if all_suite_results:
        df_suite = pd.DataFrame(all_suite_results)
        master_csv = os.path.join(output_dir, "aopc_suite_master_summary.csv")
        master_json = os.path.join(output_dir, "aopc_suite_master_summary.json")
        
        df_suite.to_csv(master_csv, index=False)

        stats = {}
        for exp, group in df_suite.groupby("experiment"):
            stats[exp] = group[["morf_aopc_score", "summary_display_aopc_score", "random_aopc_score"]].describe().to_dict()

        summary_payload = {
            "total_runs_evaluated": len(all_suite_results),
            "experiments": list(all_experiments.keys()),
            "runs": all_suite_results,
            "statistics_by_experiment": stats,
        }
        with open(master_json, "w") as f:
            json.dump(summary_payload, f, indent=4)

        LOGGER.info("=" * 80)
        LOGGER.info(f"AOPC EVALUATION SUITE COMPLETE.")
        LOGGER.info(f"Master CSV summary: {master_csv}")
        LOGGER.info(f"Master JSON summary: {master_json}")
        LOGGER.info("=" * 80)
    else:
        LOGGER.warning("No AOPC evaluation results were generated.")


if __name__ == "__main__":
    main()
