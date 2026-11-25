# From Clever Hans to Hypotheses: Interpreting EEG Foundational Transformers

This repository provides the official implementation and reproducibility framework for the paper: **"From Clever Hans to Hypotheses: Interpreting EEG Foundational Transformers"**.

📄 Paper: [arXiv:2605.11885](https://arxiv.org/abs/2605.11885)

---

## 🚀 Quick Start

We use [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

1.  **Install uv**: Follow the [official guide](https://github.com/astral-sh/uv).
2.  **Sync Environment**:
    ```bash
    uv sync
    ```
    By default this installs the standard PyTorch wheels from PyPI, which are CUDA-enabled on Linux.
    To pin a specific backend instead, select one of the mutually exclusive extras:
    ```bash
    uv sync --extra cuda126   # CUDA 12.6 wheels
    uv sync --extra cpu       # CPU-only
    uv sync --extra rocm7     # AMD ROCm 7.1
    ```
3.  **Download Checkpoints**:
    ```bash
    uv run python script/download_checkpoints.py
    ```
    This places the pretrained [LaBraM backbone](https://github.com/935963004/LaBraM/blob/main/checkpoints/labram-base.pth) and our [fine-tuned PhysioNet checkpoint](https://keeper.mpdl.mpg.de/seafhttp/f/9ebbb1d2846c48e3ba5d/?op=view) in `checkpoint/`. The script warns if a link is broken; you can also download the files manually via the links above.
4.  **Run Primary Demo (PhysioMI L/R)**:
    Automatically downloads a subset of the PhysioNet Motor Imagery dataset and runs the full pipeline (train → test → attribute):
    ```bash
    uv run python src/run_experiment.py \
      --config src/config/production/physionet_finetuned.yaml \
      --mini_n_train 10 --mini_n_val 4 --mini_n_test 4
    ```

---

## 🛠 Key Contributions

- **Validation**: We demonstrate that LRP produces plausible attributions on foundational EEG Transformers, validated against cardiac field artifacts (R-Peak detection).
- **Clever Hans Detection**: We uncover spurious reliance on eye-movement artifacts in motor imagery models (PhysioMI L/R).
- **Hypothesis Generation**: We identify a recurring sensorimotor signature in arousal prediction, surfacing novel neuroscientific hypotheses from EEG-FMs.
---

## 📂 Repository Structure

```text
├── checkpoint/             # Model weights (not tracked)
├── data/                   # Dataset storage (not tracked)
├── results/
│   ├── metrics/            # Per-run test performance of all configurations
│   ├── apa_mwu/            # Electrode statistics (Mann-Whitney U, Spearman), Table A7
│   └── aopc/               # AOPC faithfulness evaluation of run 3, Table A5
├── script/
│   ├── hyperparameter_searches/  # SLURM jobs: Optuna searches
│   ├── evaluations/        # SLURM jobs: multi-seed evaluation of tuned configs
│   ├── aopc_evaluations/   # SLURM job: AOPC faithfulness evaluation
│   ├── download_checkpoints.py
│   ├── run_aopc_eval_suite.py
│   ├── generate_aopc_report.py
│   └── quantify_attr_arousal.py
├── src/
│   ├── run_experiment.py   # Unified CLI entry point
│   ├── main.py             # Full CLI (train / test / attribute)
│   ├── tune_hparams.py     # Optuna hyperparameter search
│   ├── evaluate_model.py   # Multi-seed evaluation of a tuned study
│   ├── config/production/  # Validated configs
│   ├── models/             # LaBraM implementation (AttnLRP)
│   ├── data/               # DataModules
│   ├── evaluate/           # LRP generation, analysis utilities, AOPC
│   ├── visualization/      # Plotting utilities
│   └── notebooks/          # Analyses behind the paper's figures and tables
└── test/                   # Test suite
```

---

## 📊 Running Experiments

All experiments are quickly runnable via `src/run_experiment.py`. 
It's a reduced CLI around `main.py`. 
For full control, different hyperparameters etc. use `main.py` directly.

### A. Full Pipeline (Train → Test → Attribute)
```bash
uv run python src/run_experiment.py \
  --config src/config/production/physionet_finetuned.yaml \
  --data_dir data/physiomi
```

### B. Attribution Only (Using Published Checkpoint)
Uses the fine-tuned PhysioNet checkpoint fetched in setup step 3 (or download it [here](https://keeper.mpdl.mpg.de/seafhttp/f/9ebbb1d2846c48e3ba5d/?op=view) to `checkpoint/physionet_finetuned_best.ckpt`):
```bash
uv run python src/run_experiment.py \
  --config src/config/production/physionet_finetuned.yaml \
  --attribute_only \
  --checkpoint checkpoint/physionet_finetuned_best.ckpt
```

*Note:* Notebooks illustrate how to use the `ExperimentResults` and `AttributionAnalyzer` classes, which mainly wrap Dask-based processing of large attribution datasets.
Visualization, aggregation etc. can happen in a multitude of ways from the .parquet attribution files.

---

### C. Reproducing the Paper's Analyses

The SLURM scripts are the job files used on the HPC cluster. Paths like `/ptmp/$USER/...` refer to cluster scratch space, so adapt them to your system.
Notebooks reference run directories as `$HOME/...` or `/ptmp/$USER/...`. Point these at your own output directories.

---

## 📝 Implementation Details

- **Model**: LaBraM-base (5.8M parameters, 12 attention layers).
- **Configurations**: Support for **finetuned (ftn.)**, **frozen (frz.)**, and **from scratch (f.s.)** variants.
- **XAI**: Attention-aware LRP (AttnLRP) using the `LXT` and `Zennit` libraries.
- **Visualization**: Specialized Voronoi topographic maps and temporal relevance profiles. Full resolution interactive HTMLs with Plotly.

---

## 📜 Citations

If you use this code or our findings, please cite:

```bibtex
@inproceedings{meyerzubexten2026cleverhans,
  title={From Clever Hans to Hypotheses: Interpreting EEG Foundational Transformers},
  author={Meyer zu Bexten, Justus and Franczyk, Bogdan and Scherf, Nico and Hofmann, Simon M.},
  booktitle={NeurIPS 2026 Workshop on Foundation Models for the Brain and Body},
  year={2026},
  eprint={2605.11885},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2605.11885}
}
```

Please also cite the original LaBraM and AttnLRP works:

```bibtex
@inproceedings{jiang2024large,
  title={Large Brain Model for Learning Generic Representations with Tremendous {EEG} Data in {BCI}},
  author={Wei-Bang Jiang and Li-Ming Zhao and Bao-Liang Lu},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}

@inproceedings{achtibat2024attnlrp,
  title={{AttnLRP}: Attention-Aware Layer-Wise Relevance Propagation for Transformers},
  author={Achtibat, Reduan and Hatefi, Sayed Mohammad Vakilzadeh and Dreyer, Maximilian and Jain, Aakriti and Wiegand, Thomas and Lapuschkin, Sebastian and Samek, Wojciech},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  series={Proceedings of Machine Learning Research},
  volume={235},
  pages={135--168},
  publisher={PMLR},
  year={2024}
}
```

---

## ⚖️ License
Licensed under the MIT License. See [LICENSE](LICENSE) for details.

---
*Note: This repository is the official implementation for reproducibility. It is intended for research purposes to support the findings of our publication.*
