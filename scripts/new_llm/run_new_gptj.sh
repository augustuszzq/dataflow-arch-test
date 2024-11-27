#!/bin/bash

# Log File Path
LOG_FILE=~/mixed_precision/scripts/new_llm/setup_and_train.log

# Output error to Log
exec > >(tee -i $LOG_FILE) 2>&1

# Step 1: Clean up existing virtual environment if it exists
echo "Cleaning up any existing virtual environment..."
if [ -d "~/mixed_precision/scripts/new_llm/venv_cerebras_pt" ]; then
  rm -rf ~/mixed_precision/scripts/new_llm/venv_cerebras_pt
fi

# Step 2: Set up a new Cerebras virtual environment
echo "Setting up the Cerebras virtual environment..."
/opt/python3.8/bin/python3.8 -m venv ~/mixed_precision/scripts/new_llm/venv_cerebras_pt
source ~/mixed_precision/scripts/new_llm/venv_cerebras_pt/bin/activate
export PYTHONPATH="~/mixed_precision/scripts/new_llm/modelzoo/src:$PYTHONPATH"

# Step 3: Upgrade pip and install Cerebras PyTorch package version 2.3.0
pip install --upgrade pip
pip install cerebras_pytorch==2.3.0

# Step 4: Clone or update the modelzoo repository to the correct version
mkdir -p ~/mixed_precision/R_2.3.0
cd ~/mixed_precision/R_2.3.0
if [ ! -d "modelzoo" ]; then
  git clone https://github.com/Cerebras/modelzoo.git
  cd modelzoo
  git checkout Release_2.3.0
else
  cd modelzoo
  git pull
  git checkout Release_2.3.0
fi

# Step 5: Install required packages from Model Zoo
pip install -r ~/mixed_precision/R_2.3.0/modelzoo/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Step 6: Set up model directory
export MODEL_DIR=model_dir_gptj
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi

# Step 7: Run the training script with the YAML file from the modelzoo directory
cd ~/mixed_precision/R_2.3.0/modelzoo/src/cerebras/modelzoo/models/nlp/gptj

python run.py CSX \
  --job_labels name=gptj_pt \
  --params /home/kevienzzq/mixed_precision/R_2.3.0/modelzoo/src/cerebras/modelzoo/models/nlp/gptj/configs/params_gptj_6B.yaml \
  --num_csx=2 \
  --mode train \
  --model_dir $MODEL_DIR \
  --mount_dirs /home/ /software \
  --python_paths /home/$(whoami)/mixed_precision/R_2.3.0/modelzoo/src \
  --compile_dir $(whoami) |& tee -a $LOG_FILE

# Step 8: Clean up the virtual environment
echo "Cleaning up the virtual environment..."
deactivate
rm -rf ~/mixed_precision/scripts/new_llm/venv_cerebras_pt

echo "Script execution completed. Virtual environment removed."
