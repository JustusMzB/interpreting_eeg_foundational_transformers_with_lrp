#!/usr/bin/env python
"""
Unified entry point for running experiments.
Trains the model, evaluates on the test set, and generates LRP attributions
using a single YAML configuration. AOPC evaluation: script/run_aopc_eval_suite.py.
"""
import argparse
import logging
import sys
import os
LOGGER = logging.getLogger(__name__)

def build_parser():
    parser = argparse.ArgumentParser(description="Run a full experiment pipeline or evaluation.")
    parser.add_argument("--config", default=None, help="Path to the YAML configuration file.")
    parser.add_argument("--data_dir", default=None, help="Override data_dir in the config.")
    parser.add_argument("--output_dir", default=None, help="Override output_dir in the config.")
    parser.add_argument(
        "--attribute_only", action="store_true",
        help="Skip training; run attribution only using --checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a model checkpoint (required with --attribute_only).",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (minimal data/epochs).")
    parser.add_argument("--mini_n_train", type=int, default=None,
                        help="Mini-dataset mode (physionetmi_lr only): training subjects.")
    parser.add_argument("--mini_n_val", type=int, default=None,
                        help="Mini-dataset mode (physionetmi_lr only): validation subjects.")
    parser.add_argument("--mini_n_test", type=int, default=None,
                        help="Mini-dataset mode (physionetmi_lr only): test subjects.")
    return parser


def run_experiment(args):
    """Run the standard train → test → attribute pipeline."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from main import main as main_fn

    if not args.config:
        raise ValueError("--config argument is required for training/testing experiments.")

    argv = ["--config", args.config, "--do_testing", "--attribute"]
    if args.data_dir:
        argv += ["--data_dir", args.data_dir]
    if args.output_dir:
        argv += ["--output_dir", args.output_dir]
    if args.workers:
        argv += ["--num_workers", str(args.workers)]
    if args.debug:
        argv += ["--debug"]
    if args.mini_n_train is not None:
        argv += ["--mini_n_train", str(args.mini_n_train)]
    if args.mini_n_val is not None:
        argv += ["--mini_n_val", str(args.mini_n_val)]
    if args.mini_n_test is not None:
        argv += ["--mini_n_test", str(args.mini_n_test)]

    main_fn(argv)


def run_attribution_only(args):
    """Load a checkpoint and run attribution, skipping training."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from main import parse_args as main_parse_args, build_data_module
    from models.lighting_labram import NeuralTransformerLightning
    from evaluate.attribute import attribute_dataset_on_labram
    import shutil

    if not args.config:
        raise ValueError("--config is required for attribution-only mode.")

    main_args = main_parse_args(["--config", args.config])
    if args.data_dir:
        main_args.data_dir = args.data_dir
    if args.mini_n_train is not None:
        main_args.mini_n_train = args.mini_n_train
    if args.mini_n_val is not None:
        main_args.mini_n_val = args.mini_n_val
    if args.mini_n_test is not None:
        main_args.mini_n_test = args.mini_n_test
    output_dir = args.output_dir or main_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data module...")
    data_module = build_data_module(main_args)

    print(f"Loading model from {args.checkpoint}...")
    model = NeuralTransformerLightning.load_from_checkpoint(
        args.checkpoint, weights_only=False
    )

    val_out = os.path.join(output_dir, "attribution_results_val.parquet")
    shutil.rmtree(val_out, ignore_errors=True)
    print("Running attribution on validation set...")
    attribute_dataset_on_labram(
        data_module.val_ds,
        model,
        output_path=val_out,
        num_workers=args.workers,
    )

    if hasattr(data_module, "test_ds"):
        test_out = os.path.join(output_dir, "attribution_results_test.parquet")
        shutil.rmtree(test_out, ignore_errors=True)
        print("Running attribution on test set...")
        attribute_dataset_on_labram(
            data_module.test_ds,
            model,
            output_path=test_out,
            num_workers=args.workers,
        )

    print("Attribution-only run completed successfully.")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.attribute_only:
        if not args.config:
            parser.error("--config is required when using --attribute_only.")
        if not args.checkpoint:
            print("Error: --checkpoint is required with --attribute_only.", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint '{args.checkpoint}' not found.", file=sys.stderr)
            sys.exit(1)
        run_attribution_only(args)
    else:
        if not args.config:
            parser.error("the following arguments are required: --config")
        if not os.path.exists(args.config):
            print(f"Error: Config file '{args.config}' not found.", file=sys.stderr)
            sys.exit(1)
        run_experiment(args)


