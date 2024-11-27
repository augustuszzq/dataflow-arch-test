#!/bin/bash

#Log File Path
LOG_FILE=~/mixed_precision/scripts/cnntest/setup_and_train.log

# Output error to Log
exec > >(tee -i $LOG_FILE) 2>&1

# Step 1: Clean up existing virtual environment if it exists
echo "Cleaning up any existing virtual environment..."
if [ -d "~/mixed_precision/scripts/cnntest/venv_cerebras_pt" ]; then
  rm -rf ~/mixed_precision/scripts/cnntest/venv_cerebras_pt
fi

# Step 2: Set up a new Cerebras virtual environment
echo "Setting up the Cerebras virtual environment..."
/opt/python3.8/bin/python3.8 -m venv ~/mixed_precision/scripts/cnntest/venv_cerebras_pt
source ~/mixed_precision/scripts/cnntest/venv_cerebras_pt/bin/activate
export PYTHONPATH="~/mixed_precision/scripts/llm/modelzoo/src:$PYTHONPATH"

# Step 3: Upgrade pip and install Cerebras PyTorch package version 2.2.1
pip install --upgrade pip
pip install cerebras_pytorch==2.3.0

# Step 4: Install required packages from Model Zoo
cd ~/mixed_precision/scripts/cnntest/modelzoo
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Step 5: Run the data.py script to calculate steps_per_epoch
# echo "Running data.py to calculate steps_per_epoch..."
# python ~/mixed_precision/scripts/llm/data.py

# Step 6: Set up model directory
export MODEL_DIR=model_dir_gptj
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi

# Step 7: Run the training script with the YAML file from the modelzoo directory
python ~/mixed_precision/scripts/cnntest/modelzoo/src/cerebras/modelzoo/models/nlp/gptj/run.py \
  CSX \
  --job_labels name=gptj_pt \
  --params ~/mixed_precision/scripts/cnntest/modelzoo/src/cerebras/modelzoo/models/nlp/gptj/configs/params_gptj_6B.yaml \
  --num_csx=2 \
  --mode train \
  --model_dir $MODEL_DIR \
  --mount_dirs /home/ /software \
  --python_paths /home/kevienzzq/mixed_precision/scripts/cnntest/modelzoo/src/cerebras/modelzoo \
  --compile_dir $(whoami) |& tee -a $LOG_FILE

# Optional: Commented out alternative training script for reference
# python ~/mixed_precision/scripts/llm/modelzoo/src/cerebras/modelzoo/models/nlp/gptj/run.py \
#   CSX \
#   --params ~/mixed_precision/scripts/llm/modelzoo/src/cerebras/modelzoo/models/nlp/gptj/configs/params_gptj_6B.yaml \
#   --num_csx=1 \
#   --model_dir train_from_scratch_GPT111M \
#   --mode train \
#   --mount_dirs /software/cerebras/dataset/gptj/train_8M_msl2048 \
#   --python_paths ~/mixed_precision/scripts/llm/modelzoo/src \
#   --job_labels model=GPT111M

# Step 8: Clean up the virtual environment
echo "Cleaning up the virtual environment..."
deactivate
rm -rf ~/mixed_precision/scripts/cnntest/venv_cerebras_pt

echo "Script execution completed. Virtual environment removed."
