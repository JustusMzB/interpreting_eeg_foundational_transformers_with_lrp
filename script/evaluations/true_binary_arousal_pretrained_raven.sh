#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./out/%x.out.%j
#SBATCH -e ./out/%x.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J true_binary_pretrained_arousal_evaluation
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
 --base_config src/config/production/arousal_finetuned.yaml \
 --study_dir /ptmp/$USER/logs/true_binary_arousal_tuning_pretrained\
 --study_name "true_binary_arousal_tuning_pretrained" \
 --sweep_yaml src/config/sweeps/affect_sweep_v2.yaml \
 --n_runs 5 \
 --output_dir /ptmp/$USER/logs/true_binary_arousal_pretrained_evaluation \
 --num_worker_override 14