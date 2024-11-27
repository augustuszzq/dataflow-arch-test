#!/bin/bash

# Step 1: Clean up existing virtual environment if it exists
echo "Cleaning up any existing virtual environment..."
if [ -d "~/mixed_precision/scripts/venv_cerebras_pt" ]; then
  rm -rf ~/mixed_precision/scripts/venv_cerebras_pt
fi

# Step 2: Set up a new Cerebras virtual environment
echo "Setting up the Cerebras virtual environment..."
/opt/python3.8/bin/python3.8 -m venv ~/mixed_precision/scripts/venv_cerebras_pt
source ~/mixed_precision/scripts/venv_cerebras_pt/bin/activate
export PYTHONPATH="~/mixed_precision/scripts/modelzoo/src:$PYTHONPATH"

# Step 3: Upgrade pip and install Cerebras PyTorch package version 2.2.0
pip install --upgrade pip
pip install cerebras_pytorch==2.3.0

Step 4: Clone Cerebras Model Zoo
echo "Cloning the Cerebras Model Zoo..."
cd ~/mixed_precision/scripts
if [ -d "modelzoo" ]; then
 rm -rf modelzoo
fi
git clone https://github.com/Cerebras/modelzoo.git
cd modelzoo
git checkout Release_2.3.0

# Step 5: Install additional packages from the Model Zoo
pip install -r ~/mixed_precision/scripts/modelzoo/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Step 6: Set permissions for Model Zoo directory and files
echo "Setting read/write/execute permissions for Model Zoo files..."
chmod -R u+rwx ~/mixed_precision/scripts/modelzoo

# Step 7: Navigate to your model implementation directory
cd ~/mixed_precision/scripts/modelzoo/src/cerebras/modelzoo/fc_mnist/pytorch

# Print the current directory and list all files to confirm presence
echo "Current directory before running:"
pwd
echo "Listing all files in the current directory:"
ls -l

# Ensure file permissions are correct
chmod +x *.py

# Prepare the environment for running the model
# echo "Preparing datasets..."
# python prepare_data.py

echo "Launching the training job..."
MODEL_DIR=~/mixed_precision/output/model_dir_mnist
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi

# Using the local params.yaml file
LOCAL_PARAMS=~/mixed_precision/scripts/params.yaml

if [ -f "$LOCAL_PARAMS" ]; then
  echo "Using local params.yaml file: $LOCAL_PARAMS"
else
  echo "Local params.yaml file not found: $LOCAL_PARAMS"
  exit 1
fi

python run.py CSX \
  --params $LOCAL_PARAMS \
  --num_csx=1 \
  --mode train \
  --model_dir $MODEL_DIR \
  --mount_dirs /home/$(whoami)/ /software \
  --python_paths /home/$(whoami)/mixed_precision/scripts/modelzoo/src \
  --compile_dir /$(whoami) |& tee ~/mixed_precision/scripts/with_mixed_precision_100.log

echo "Training job has been launched. Check the outputs in the designated output directory."

cd ~/mixed_precision/scripts/

# Step 8: Clean up the virtual environment
echo "Cleaning up the virtual environment..."
deactivate
rm -rf ~/mixed_precision/scripts/venv_cerebras_pt

echo "Script execution completed. Virtual environment removed."
