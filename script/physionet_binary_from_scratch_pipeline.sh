#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./out/%x.out.%j
#SBATCH -e ./out/%x.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J binary_physionet_from_scratch_pipeline
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --time=24:00:00

module purge
module load python-waterboa/2024.06

# conda activate ./.conda # Activate the conda environment # replaced by uv run
export HF_HOME="/ptmp/$USER/huggingface" # Set the Hugging Face cache directory to scratch space

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
STUDY_DIR="/ptmp/$USER/logs/physionet_binary_from_scratch"
STUDY_NAME="binary_physionet_from_scratch_tuning"
BASE_CONFIG="src/config/production/physionet_from_scratch.yaml"
SWEEP="src/config/sweeps/balanced_sweep_v1.yaml"


srun uv run --extra cuda126 python src/tune_hparams.py \
 --base_config $BASE_CONFIG \
 --study_dir $STUDY_DIR \
 --study_name $STUDY_NAME \
 --sweep_yaml $SWEEP \
 --n_trials 100 \
 --warmup_epochs 40 # Fewer samples in the train split than in brain heart emotion

srun uv run src/evaluate_model.py \
 --base_config $BASE_CONFIG \
 --study_dir $STUDY_DIR\
 --study_name $STUDY_NAME \
 --sweep_yaml $SWEEP \
 --n_runs 5 \
 --output_dir /ptmp/$USER/logs/true_binary_physionet_from_scratch_evaluation
