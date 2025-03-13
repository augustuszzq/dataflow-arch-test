#!/bin/bash
# train.sh - Training script for Cerebras Modelzoo 1.6.0 using pre-cloned modelzoo
#
# Usage:
#   ./train.sh compile    # Compile the model only (--compile_only)
#   ./train.sh train      # Train the model
#
# Notes:
# 1. Make sure you have started an interactive session and launched the container beforehand:
#    srun --pty --cpus-per-task=28 --kill-on-bad-exit singularity shell --cleanenv \
#         --bind /local1/cerebras/data,/local2/cerebras/data,/local3/cerebras/data,/local4/cerebras/data,/jet/home/zzhong2/modelzoo \
#         /local1/cerebras/cbcore_latest.sif
#
# 2. The modelzoo is assumed to be located at: /jet/home/zzhong2/modelzoo/modelzoo

# set -e

# if [ "$#" -ne 1 ]; then
#   echo "Usage: $0 [compile|train]"
#   exit 1
# fi

MODE=$1

# Set the Modelzoo directory
MODELZOO_DIR="/jet/home/zzhong2/modelzoo/modelzoo"

# Define the model directory for fc_mnist (adjust if using a different model)
MODEL_DIR="$MODELZOO_DIR/transformers/pytorch/gpt2"
if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory $MODEL_DIR does not exist. Please check the modelzoo path."
  exit 1
fi

cd "$MODEL_DIR"

# Compile or train the model based on the mode argument
if [ "$MODE" = "compile" ]; then
  echo "Compiling the model..."
  python-pt run.py --mode train -p configs/params_gpt2_small.yaml --compile_only --model_dir compile
elif [ "$MODE" = "train" ]; then
  echo "Starting model training..."
  python-pt run.py --mode train -p configs/params_gpt2_small.yaml --model_dir train_output
else
  echo "Unknown mode: $MODE. Please use either compile or train."
  exit 1
fi
