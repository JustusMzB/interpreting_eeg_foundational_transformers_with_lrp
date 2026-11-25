#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./out/%x.out.%j
#SBATCH -e ./out/%x.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J true_binary_frozen_evaluation
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --time=6:00:00

module purge
module load python-waterboa/2024.06
eval "$(conda shell.bash hook)"
conda deactivate
conda activate ./.conda
# This should show the path to your activated conda environment
echo "✅ Conda environment: $CONDA_PREFIX" 
# print python interpreter path
echo "Using Python interpreter: $(which python)"

# Print optuna and torch versions
echo "Using Optuna versions:" 
echo "$(conda list | grep optuna | awk '{print $1} {print $2}')"
echo "Using PyTorch versions:"
echo "$(conda list | grep torch | awk '{print $1} {print $2}')"
# Print Python version
echo $PYTHONPATH
echo $PATH
export HF_HOME="/ptmp/$USER/huggingface" # Set the Hugging Face cache directory to scratch space

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python src/evaluate_model.py \
 --base_config src/config/experimental/arousal_frozen_binary_mode.yaml \
 --study_dir /ptmp/$USER/logs/arousal_frozen_binary_mode_v3 \
 --study_name "arousal_frozen_binary_mode_v3" \
 --sweep_yaml src/config/sweeps/affect_sweep_frozen.yaml \
 --output_dir /ptmp/$USER/logs/arousal_frozen_crossval_evaluation \
 --num_worker_override 14\
 --kfold 5