#!/usr/bin/env python
#print("peep, I am alive.")
"""
train.py

Training script for NeuralTransformerLightning, adapted from run_class_finetuning.py,
with modular functions for easy Optuna integration, full feature coverage,
YAML-based configuration, and debug mode for quick test runs on a laptop.
"""
import warnings

from data.physionet_datamodule import PhysionetDatamodule

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

import argparse
import glob
import os
import random
import json
import yaml
import numpy as np
import torch
import shutil
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichModelSummary
from models.lighting_labram import NeuralTransformerLightning
from data.affect_prediction_datamodule import AffectPredictionDataModule
from data.is_rpeak_datamodule import IsRpeakPredictionDataModule
from evaluate.attribute import attribute_dataset, attribute_dataset_on_labram
import sys
from optuna.integration import PyTorchLightningPruningCallback
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import logging
from typing import Literal
import multiprocessing


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Lightning fine-tuning for LaBraM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # YAML config file
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file. CLI args override YAML settings.'
    )
    # Debug / test mode
    parser.add_argument('--debug', action='store_true',
                        help='Activate debug mode: minimal data, epochs, and no checkpointing.')
    # Eval arguments
    parser.add_argument('--do_testing', action='store_true',
                        help='Run evaluation on test set after training. Default is False.')
    parser.add_argument('--attribute', action='store_true',
                        help='Run attribution on validation and test set after training. Default is False.')
    parser.add_argument('--attributions_per_example', type=int, default=None,
                        help='Number of attributions to compute per example. Default is None (all). The attributions file will be the size of the data attributed multiplied by this.')
    # Basics
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--update_freq', type=int, default=1)
    #parser.add_argument('--save_ckpt_freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto')

    # Model selection
    parser.add_argument('--model_type', choices=['base', 'large', 'huge'], default='base')
    parser.add_argument('--EEG_size', type=int, default=1600)
    parser.add_argument('--patch_size', type=int, default=200)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_chans', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=200)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=10)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--qkv_bias', action='store_true', default=False)
    parser.add_argument('--qk_norm', type=str, default='LayerNorm')
    parser.add_argument('--qk_scale', type=float, default=None)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--init_values', type=float, default=0.1)
    parser.add_argument('--use_abs_pos_emb', action='store_true', default=True)
    parser.add_argument('--use_rel_pos_bias', action='store_true', default=False)
    parser.add_argument('--use_shared_rel_pos_bias', action='store_true', default=False)
    parser.add_argument('--use_mean_pooling', action='store_true', default=True)
    parser.add_argument('--init_scale', type=float, default=0.001)

    # Optimization
    parser.add_argument('--optimizer', choices=['adamw', 'sgd'], default='adamw')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    # Regularization
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--no_early_stopping', action='store_false', dest='early_stopping', default=True, help='Disable early stopping based on validation loss.')

    # Loss function
    parser.add_argument('--loss_fn', choices=["cross_entropy", "mse"], default="cross_entropy")
    parser.add_argument('--class_weights', nargs='+', type=float, default=None)

    # LR schedule
    parser.add_argument('--min_lr', type=float, default=0)




    # Fine-tuning
    parser.add_argument('--finetune_ckpt', type=str, default=None)
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    parser.add_argument('--head_n_hidden_layers', type=int, default=0, help='Number of hidden layers in the head. Default is 0 (linear probing).')
    parser.add_argument('--head_hidden_dim', type=int, default=16, help='Hidden dimension of the head. Default is 200.')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warmup epochs for the learning rate scheduler. Required mostly for FTn')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-5, help='Starting learning rate for the warmup phase. Default is 1e-5.')
    # Data
    parser.add_argument('--dataset', 
                        choices=[
                            'brain_heart_rpeak', 
                            'brain_heart_arousal', 
                            'brain_heart_arousal_tertiary',
                            'brain_heart_valence',
                            'brain_heart_valence_long',
                            'brain_heart_arousal_long',
                            'brain_heart_valence_tertiary',
                            'physionetmi_lr'
                            ], default='brain_heart_arousal')
    parser.add_argument('--label_type', choices=['patchwise', 'seq2seq'], default='patchwise',) # Affects model init, is determined by dataset.
    parser.add_argument('--dataset_path', type=str, default='data/brain_heart_emotion')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--mini_n_train', type=int, default=None,
                        help='Mini-dataset mode (physionetmi_lr only): number of training subjects to download/use.')
    parser.add_argument('--mini_n_val', type=int, default=None,
                        help='Mini-dataset mode (physionetmi_lr only): number of validation subjects.')
    parser.add_argument('--mini_n_test', type=int, default=None,
                        help='Mini-dataset mode (physionetmi_lr only): number of test subjects.')
    parser.add_argument('--kfold', type=int, default=None, help='Number of folds for k-fold cross-validation. If set, will use k-fold cross-validation.')
    parser.add_argument('--fold_index', type=int, default=None, help='Index of the fold to use for testing.')
    parser.add_argument('--kfold_seed', type=int, default=16081832, help='Random seed to make sure same folds are used.')
    parser.add_argument('--exclude_electrodes', type=str, default=None, help='Comma-separated list of electrode names to exclude from input.')
    parser.add_argument('--include_electrodes', type=str, default=None, help='Comma-separated list of electrode names to include (all others excluded).')
    parser.add_argument('--ablation_strategy', type=str, choices=['drop', 'zero'], default='drop', help='Ablation strategy: drop channels or zero out their values.')


    parser.add_argument('--nb_classes', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='out')
    parser.add_argument('--no_resume', action='store_true', default=False)
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--profile', action='store_true', default=False)

    # Hyperparameter tuning: The main is run as part of an optuna trial
    parser.add_argument('--optuna_tuning', action='store_true', default=False)
    parser.add_argument('--target_metric', type=str, default="val_balanced_accuracy")
    parser.add_argument('--target_metric_direction', type=str, default="max")
    parser.add_argument('--task_type', choices=['multiclass', 'binary'], default=None, help='Task type for metrics. Inferred from nb_classes if not set.')

    # logging
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')

    if argv is None:
        argv = sys.argv[1:]
    # First parse to load config file
    args, remaining = parser.parse_known_args(argv)
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    # Parse again with YAML defaults, CLI overrides
    args = parser.parse_args(argv)
    return args
    

def build_loss_fn(args):
    if args.loss_fn == "cross_entropy":
        weights = torch.tensor(args.class_weights) if args.class_weights else None
        if args.task_type == 'binary' and args.nb_classes == 2:
            # For binary classification, use BCEWithLogitsLoss
            loss_fn_before_smoothing = torch.nn.BCEWithLogitsLoss(
                pos_weight=weights if weights is not None else None,
            )
            # manually add label smoothing in front
            def loss_fn(preds, targets):
                # Apply label smoothing
                loss_fn_before_smoothing.to(preds.device)
                smoothing = args.smoothing
                smoothing = torch.tensor(smoothing, device=preds.device)
                targets = torch.clamp(targets.float(), min=smoothing, max=1- smoothing)
                # Apply weights if
                return loss_fn_before_smoothing(preds, targets)
            
        else:
            loss_fn = torch.nn.CrossEntropyLoss(
                label_smoothing=args.smoothing,
                weight=weights,
            )
    elif args.loss_fn == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")
    return loss_fn

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args):
    # Map string to qk_norm class
    qk_norm = getattr(torch.nn, args.qk_norm) if args.qk_norm else None
    
    if args.task_type is None:
        args.task_type = 'binary' if args.nb_classes == 2 else 'multiclass'
        logging.info(f"Inferred task type: {args.task_type}")

    model_config = {
        'EEG_size': args.EEG_size,
        'patch_size': args.patch_size,
        'in_channels': args.in_channels,
        'out_chans': args.out_chans,
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'num_heads': args.num_heads,
        'mlp_ratio': args.mlp_ratio,
        'qkv_bias': args.qkv_bias,
        'qk_norm': qk_norm,
        'qk_scale': args.qk_scale,
        'drop_rate': args.drop_rate,
        'attn_drop_rate': args.attn_drop_rate,
        'drop_path_rate': args.drop_path_rate,
        'norm_layer': torch.nn.LayerNorm,
        'init_values': args.init_values,
        'use_abs_pos_emb': args.use_abs_pos_emb,
        'use_rel_pos_bias': args.use_rel_pos_bias,
        'use_shared_rel_pos_bias': args.use_shared_rel_pos_bias,
        'use_mean_pooling': args.use_mean_pooling,
        'init_scale': args.init_scale,
        'label_type': args.label_type,  # Affects model init, is determined by dataset.
        'head_hidden_dim': args.embed_dim,  # Default to embed_dim
        'head_n_hidden_layers': args.head_n_hidden_layers,  # Default to 1 hidden layer in the head
        'head_hidden_dim': args.head_hidden_dim,  # Default to 16
    }
    
    loss_fn = build_loss_fn(args)

    model = NeuralTransformerLightning(
        model_config=model_config,
        num_classes=args.nb_classes,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_fn=loss_fn,
        pretrained_path=args.finetune_ckpt,
        output_modality=args.label_type,
        freeze_backbone=args.freeze_backbone,
        task_type=args.task_type,
        track_gradients=args.debug,
        warmup_epochs=args.warmup_epochs,
        warmup_start_lr=args.warmup_start_lr,
        min_lr=args.min_lr,
    )   
    return model


def build_data_module(args):
    mini_flags = (
        getattr(args, 'mini_n_train', None),
        getattr(args, 'mini_n_val', None),
        getattr(args, 'mini_n_test', None),
    )
    if any(v is not None for v in mini_flags) and args.dataset != 'physionetmi_lr':
        raise ValueError(
            f"--mini_n_train/val/test are only supported for dataset 'physionetmi_lr', "
            f"got '{args.dataset}'."
        )

    ablation_kwargs = {}
    if getattr(args, 'exclude_electrodes', None):
        ablation_kwargs['exclude_electrodes'] = args.exclude_electrodes
    if getattr(args, 'include_electrodes', None):
        ablation_kwargs['include_electrodes'] = args.include_electrodes
    if getattr(args, 'ablation_strategy', None):
        ablation_kwargs['ablation_strategy'] = args.ablation_strategy

    datamodule_map = {
        'brain_heart_arousal_tertiary': lambda *args, **kwargs: AffectPredictionDataModule(*args, **kwargs, emotional_dimension="arousal", num_classes=3, **ablation_kwargs),
        'brain_heart_arousal': lambda *args, **kwargs: AffectPredictionDataModule(*args, **kwargs, emotional_dimension="arousal", num_classes=2, **ablation_kwargs),
        'brain_heart_valence': lambda *args, **kwargs: AffectPredictionDataModule(*args, **kwargs, emotional_dimension="valence", num_classes=2, **ablation_kwargs),
        'brain_heart_valence_tertiary': lambda *args, **kwargs: AffectPredictionDataModule(*args, **kwargs, emotional_dimension="valence", num_classes=3, **ablation_kwargs),
        'brain_heart_arousal_long': lambda *args, **kwargs: AffectPredictionDataModule(*args, **kwargs, dataset_name="brain_heart_emotion_long", emotional_dimension="arousal", num_classes=2, **ablation_kwargs),
        'brain_heart_valence_long': lambda *args, **kwargs: AffectPredictionDataModule(*args, **kwargs, dataset_name="brain_heart_emotion_long", emotional_dimension="valence", num_classes=2, **ablation_kwargs),
        'brain_heart_rpeak': IsRpeakPredictionDataModule,
        'physionetmi_lr': lambda *args, **kwargs: PhysionetDatamodule(
            *args, **kwargs,
            classes=["left_hand", "right_hand"],
            mini_n_train=mini_flags[0],
            mini_n_val=mini_flags[1],
            mini_n_test=mini_flags[2],
            **ablation_kwargs
        )
    }
    # Load dataset
    data_module = datamodule_map[args.dataset](
        dataset_path=args.dataset_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        load_dataset_kwargs={
                'kfold_k': args.kfold,
                'kfold_seed': args.kfold_seed,
                'kfold_index': args.fold_index
        }
    )

    data_module.prepare_data()
    data_module.setup()

    if hasattr(data_module, 'keep_indices') and (getattr(args, 'exclude_electrodes', None) or getattr(args, 'include_electrodes', None)):
        if getattr(args, 'ablation_strategy', 'drop') == 'drop':
            args.in_channels = len(data_module.keep_indices)
            logging.info(f"Electrode ablation (drop): Updated in_channels to {args.in_channels} (Kept: {len(data_module.keep_indices)}, Excluded: {len(data_module.excluded_indices)})")
        else:
            logging.info(f"Electrode ablation (zero): in_channels remains {args.in_channels} (Zeroed: {len(data_module.excluded_indices)} electrodes)")

    return data_module


def build_trainer(args, trial=None):
    # Logger & callbacks
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir, 
        name=args.dataset,
        version=None if args.no_resume else 0,  # Use None to create a new version)
    )
    logging.info(f"Logging to {tb_logger.log_dir}")
    target_metric_callback = ModelCheckpoint(
        monitor=args.target_metric,
        mode=args.target_metric_direction,
        dirpath=tb_logger.log_dir,
        filename = '{' + str(args.target_metric) + ':.2f}-{epoch:02d}-{val_loss:.2f}-target_best',
        save_top_k=1 if not args.debug else 0,
        #every_n_epochs=args.save_ckpt_freq,
        save_last=True,
        verbose=True,
    )
    val_loss_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=tb_logger.log_dir,
        filename = '{val_loss:.2f}-{epoch:02d}-loss_best',
        save_top_k=1 if not args.debug else 0,
        #every_n_epochs=args.save_ckpt_freq,
        save_last=True,
    )
    info_callback = RichModelSummary(
        max_depth=3,
        show_name=True,
        show_input=True,
        show_output=True,
    )
    callbacks = [
        info_callback,
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(
            monitor='val_loss',
            patience=args.epochs // 10 if args.early_stopping else args.epochs,  # If early stopping is disabled, set patience to number of epochs to effectively disable it
            verbose=True,
            mode='min',
        ),
        target_metric_callback,
        val_loss_callback,
    ]
        
    if args.optuna_tuning and trial:
        logging.info(f'Optuna tuning enabled. Adding pruning callback.')
        logging.debug(f'Optuna trial number: {trial.number}')
        logging.debug(f'Optuna trial params: {trial.params}')
        logging.debug(f'Monitoring metric: {args.target_metric}')
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor=args.target_metric))

    trainer = Trainer(
        max_epochs= args.epochs,
        accelerator=args.device,
        devices='auto',
        strategy='auto',
        precision='16-mixed',
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps=10 if not args.debug else 1,
        accumulate_grad_batches=args.update_freq,
        limit_train_batches=5 if args.debug else None,
        limit_val_batches=3 if args.debug else None,
        limit_test_batches=3 if args.debug else None,
        profiler="simple" if args.profile else None,
        enable_progress_bar= not args.optuna_tuning,
    )
    return trainer, target_metric_callback

def determine_state(log_dir) -> Literal["not_started", "training", "trained_untested", "tested", "tested_attributed", "attributed_val_only"]:
    """
    Determine the state of the training run based on the presence of certain files.
    """
    if not os.path.exists(log_dir):
        return "not_started"
    ### Check for later files first
    elif os.path.exists(os.path.join(log_dir, "test_results.json")):
    
        if os.path.exists(os.path.join(log_dir, "attribution_results_test.parquet")):
            return "tested_attributed"
        else:
            return "tested"
    elif os.path.exists(os.path.join(log_dir, "attribution_results_val.parquet")):
        # If attribution on validation set is done, but no testing took place
        return "attributed_val_only"
    
    elif os.path.exists(os.path.join(log_dir, "last.ckpt")):
        return "trained_untested"
    ### Logdir found, but no final checkpoint or test results
    elif len(glob.glob(os.path.join(log_dir, "*loss_best.ckpt"))) > 0:
        return "training"
    else:
        # Though logdir exists, no advanced files do.
        return "not_started"

def main(argv=None, trial=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        handlers=[logging.StreamHandler(sys.stdout),],
    )
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed:
        set_seed(args.seed)

    if args.num_workers > multiprocessing.cpu_count():
        logging.warning(f"Number of workers ({args.num_workers}) exceeds available CPU cores ({multiprocessing.cpu_count()}). Setting num_workers to {multiprocessing.cpu_count()}.")
        args.num_workers = os.cpu_count()
    if args.debug:
        logging.info("[DEBUG] Running in debug mode: minimal settings enabled.")
        args.epochs = 2
        args.batch_size = min(args.batch_size, 8)
        # args.num_workers = 1
        args.pin_mem = False
        args.save_ckpt_freq = 1
        args.update_freq = 1

    data_module = build_data_module(args)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader() if args.do_testing else None

    model = build_model(args)
    trainer, target_callback = build_trainer(args, trial)
    torch.set_float32_matmul_precision('high') # Set high precision for matmul operations for tpu's
    ### State determination
    state = determine_state(trainer.logger.log_dir)
    chkpt_path = None # Initialize state dependent variable
    logging.info(f"Current state of the run: {state}")

    ### Resume  or start training
    if state in ["not_started", "training"]:
        if state == "training":
            checkpoint_paths = glob.glob(os.path.join(trainer.logger.log_dir, "*loss_best.ckpt"))
            assert len(checkpoint_paths) == 1, f"Expected exactly one checkpoint file, found {len(checkpoint_paths)}. Please check your training run."
            logging.info(f"Resuming from checkpoint: {checkpoint_paths[0]}")
            chkpt_path = checkpoint_paths[0]
        trainer.fit(
            model, 
            train_loader, 
            val_loader,
            ckpt_path=chkpt_path,
            weights_only=False,
            )
        logging.info(f"Best model path: {target_callback.best_model_path}")
        logging.info(f"Best model score ({args.target_metric}): {target_callback.best_model_score}")
        chkpt_path = target_callback.best_model_path
        state= "trained_untested"

    
    ### Load checkpoint for evaluation or attribution if already trained
    if not chkpt_path:
        logging.info(f"Loading best model from checkpoint: {target_callback.best_model_path}")
        # Find the last checkpoint
        pattern = os.path.join(trainer.logger.log_dir, "*target_best.ckpt")
        checkpoint_paths = glob.glob(pattern)
        if not args.debug:
            assert len(checkpoint_paths) == 1, f"Expected exactly one checkpoint file, found {len(checkpoint_paths)} using {pattern}. Please check your training run."
            chkpt_path = checkpoint_paths[0]
        elif len(checkpoint_paths) != 1:
            logging.warning(f'[DEBUG] no correct amount of checkpoints found, continuing with current model state for debug purposes.')
        
    
    ### Evaluate on test set if requested and not already done
    if test_loader is not None and args.do_testing and state not in ["tested", "tested_attributed"]:
        logging.info("--- Starting evaluation on test set ---")
        logging.info(f"Trainer steps before loading the model: {trainer.global_step}")
        #model = model.load_from_checkpoint(
        #    chkpt_path,
        #)
        #logging.info(f"Loaded best model from {chkpt_path}")
        if not chkpt_path:
            logging.warning("No best model found. evaluating with current model state.")
            if not args.debug:
                # Fail if this is a real run to save resources
                raise ValueError("No best model found. Please check your training run.")
            chkpt_path = None
        
        test_results = trainer.test(
            model=model, 
            dataloaders=test_loader, 
            ckpt_path=chkpt_path,
            weights_only=False,
        )
        logging.info(f'Global step after testing and resuming from checkpoint: {trainer.global_step}')
        test_results = trainer.callback_metrics
        for key, value in test_results.items():
            if isinstance(value, torch.Tensor):
                test_results[key] = value.item()
        logging.info("Test results:")
        logging.info(test_results)
        # Save results to a JSON file
        results_path = os.path.join(trainer.logger.log_dir, "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=4)
        logging.info(f"Test results saved to {results_path}")
        state = "tested"
    
    ### Run attribution on validation and test set if requested
    if args.attribute and state not in ["tested_attributed"]:
        # Only checks that no attribution on the test set has been done.
        # Would repeat attribution on the validation set if it was already done.
        # DISCLAIMER: MUST BE AFTER ANY ANY BACKPROP ON THE MODEL
        # LXT MONKEY PATCHING WOULD BREAK GRADIENTS
        assert hasattr(data_module, 'val_ds'), "Data module must have a validation dataset for attribution."
        logging.info("Starting attribution on validation dataset.")
        if chkpt_path is None:
            try :
                chkpt_path = target_callback.best_model_path
            except RuntimeError as e:
                logging.error(f"Error loading best model path: {e}")
                chkpt_path = None
        
            if not args.debug:
                # Fail if this is a real run to save resources
                raise ValueError("No best model found. Please check your training run.")
            logging.warning("No best model found. Using current model state for attribution.")
        else:
            model = NeuralTransformerLightning.load_from_checkpoint(
                chkpt_path,
                weights_only=False,
            )
        logging.info(f'Resuming attribution with model from {chkpt_path}.')
        logging.info(f'clearing previous validation attributions, assuming them to be invalid.')
        shutil.rmtree(os.path.join(trainer.logger.log_dir, "attribution_results_val.parquet"), ignore_errors=True)
        attribute_dataset_on_labram(
            data_module.val_ds if not args.debug else data_module.val_ds.take(10), 
            model, 
            output_path=os.path.join(trainer.logger.log_dir, "attribution_results_val.parquet"),
            num_workers=args.num_workers,
            subsample_time_steps= args.attributions_per_example,  # Number of attributions per example
        )

        if args.do_testing:
            logging.info("Attribute model on test set.")
            assert hasattr(data_module, 'test_ds'), "Data module must have a test dataset for attribution."
            logging.info(f'clearing previous test attributions, assuming them to be invalid.')
            shutil.rmtree(os.path.join(trainer.logger.log_dir, "attribution_results_test.parquet"), ignore_errors=True)
            attribute_dataset_on_labram(
                data_module.test_ds if not args.debug else data_module.test_ds.take(10), 
                model, 
                output_path=os.path.join(trainer.logger.log_dir, "attribution_results_test.parquet"),
                num_workers=args.num_workers,
                subsample_time_steps=args.attributions_per_example,  # Number of attributions per example
            )

    return target_callback.best_model_score


if __name__ == '__main__':
    main()
