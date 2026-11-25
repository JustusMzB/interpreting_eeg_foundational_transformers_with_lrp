#!/bin/bash -l
#SBATCH -o ./out/%x.out.%j
#SBATCH -e ./out/%x.err.%j
#SBATCH -D ./
#SBATCH -J true_binary_rpeak_pretrained_evaluation
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=6:00:00

module purge
module load python-waterboa/2024.06
eval "$(conda shell.bash hook)"
conda deactivate
conda activate ./.conda
echo "✅ Conda environment: $CONDA_PREFIX"
echo "Using Python interpreter: $(which python)"
echo "Using Optuna versions:"
echo "$(conda list | grep optuna | awk '{print $1} {print $2}')"
echo "Using PyTorch versions:"
echo "$(conda list | grep torch | awk '{print $1} {print $2}')"
echo $PYTHONPATH
echo $PATH
export HF_HOME="/ptmp/$USER/huggingface"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python src/evaluate_model.py \
 --base_config src/config/production/rpeak_finetuned.yaml \
 --study_dir /ptmp/$USER/logs/rpeak_true_binary_pretrained_tuning \
 --study_name "true_binary_rpeak_tuning_pretrained" \
 --sweep_yaml src/config/sweeps/rpeak_sweep_v1.yaml \
 --n_runs 5 \
 --output_dir /ptmp/$USER/logs/true_binary_rpeak_pretrained_evaluation \
 --num_worker_override 14\
 --subsample_attributions 4\