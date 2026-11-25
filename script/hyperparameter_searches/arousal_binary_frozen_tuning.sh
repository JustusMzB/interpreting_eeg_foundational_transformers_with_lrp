#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./out/%x.out.%j
#SBATCH -e ./out/%x.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J hyperparameter_tuning_arousal_frozen
#
#SBATCH --ntasks=1
#SBATCH --constraint="apu"
#
# --- default case: use a single APU on a shared node ---
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120000
#
# --- uncomment to use 2 APUs on a full node ---
# #SBATCH --gres=gpu:2
# #SBATCH --cpus-per-task=48
# #SBATCH --mem=240000
#

#SBATCH --time=24:00:00

module purge
module load python-waterboa/2024.06

conda activate ./.conda # Activate the conda environment
export HF_HOME="/ptmp/$USER/huggingface" # Set the Hugging Face cache directory to scratch space

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python src/tune_hparams.py \
 --base_config src/config/experimental/arousal_frozen_binary_mode.yaml \
 --study_dir /ptmp/$USER/logs/arousal_frozen_binary_mode_v3 \
 --study_name "arousal_frozen_binary_mode_v3" \
 --n_trials 100 \
 --warmup_epochs 10 \
 --sweep_yaml src/config/sweeps/affect_sweep_frozen.yaml