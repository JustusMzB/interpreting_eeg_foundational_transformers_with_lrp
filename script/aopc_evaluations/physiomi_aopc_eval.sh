#!/bin/bash
# Standard output and error:
#SBATCH -o ./out/%x.out.%j
#SBATCH -e ./out/%x.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J physiomi_aopc_eval
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --time=24:00:00

module purge
module load python-waterboa/2024.06

export HF_HOME="/ptmp/$USER/huggingface"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

BASE_DIR="/ptmp/$USER/logs"
OUTPUT_DIR="/ptmp/$USER/logs/physiomi_aopc_eval_results"

echo "Starting PhysioMI AOPC Evaluation Suite..."

srun uv run script/run_aopc_eval_suite.py \
  --suite physiomi \
  --base_dir ${BASE_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --aopc_perturbation_strategy global_mean \
  --global_ordering magnitude \
  --per_sample_ordering prediction \
  --aopc_max_steps 10 \
  --aopc_split test \
  --workers ${SLURM_CPUS_PER_TASK}

echo "PhysioMI AOPC Evaluation complete."
