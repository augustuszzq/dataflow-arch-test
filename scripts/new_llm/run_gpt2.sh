#!/bin/bash
# Log File Path (keeping your original path)
LOG_FILE=~/mixed_precision/scripts/new_llm/gpt2_train_test_cb16.log
# Redirect output to log file
exec > >(tee -i $LOG_FILE) 2>&1
echo "Starting GPT-2 training script at $(date)"
# Clean up existing virtual environment if it exists
# echo "Cleaning up any existing virtual environment..."
# if [ -d "~/mixed_precision/scripts/new_llm/venv_cerebras_pt" ]; then
#     rm -rf ~/mixed_precision/scripts/new_llm/venv_cerebras_pt
# fi

# Set up Python virtual environment (keeping your original path)
echo "Setting up Python virtual environment..."
/opt/python3.8/bin/python3.8 -m venv ~/venv_cerebras_pt
source ~/venv_cerebras_pt_R/bin/activate

# Install required packages
# echo "Installing required packages..."
# pip install --upgrade pip

# Clone or update ModelZoo repository (keeping your original path)
mkdir -p ~/mixed_precision/R_2.3.0
cd ~/mixed_precision/R_2.3.0
if [ ! -d "modelzoo" ]; then
    echo "Cloning ModelZoo repository..."
    git clone https://github.com/Cerebras/modelzoo.git
    cd modelzoo
    git checkout Release_2.3.0
else
    echo "ModelZoo repository exists, updating..."
    cd modelzoo
    git pull
    git checkout Release_2.3.0
fi

# # Install ModelZoo requirements
# echo "Installing ModelZoo requirements..."
# pip install -r ~/mixed_precision/R_2.3.0/modelzoo/requirements.txt 

# Set up model directory (keeping your original name)
export MODEL_DIR=model_dir_gpt2_train_test_cb16
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi

# Run training
echo "Starting training..."
cd ~/mixed_precision/R_2.3.0/modelzoo/src/cerebras/modelzoo/models/nlp/gpt2

# Launch training job (keeping your original configuration)
python run.py CSX \
    --job_labels name=gpt2 \
    --params /home/kevienzzq/mixed_precision/R_2.3.0/modelzoo/src/cerebras/modelzoo/models/nlp/gpt2/configs/params_gpt2_small.yaml \
    --num_csx=1 \
    --mode train \
    --model_dir $MODEL_DIR \
    --mount_dirs /home/ /software \
    --python_paths /home/$(whoami)/mixed_precision/R_2.3.0/modelzoo/src \
    --compile_dir $(whoami) |& tee -a $LOG_FILE

# Cleanup
echo "Cleaning up..."
deactivate
# rm -rf ~/mixed_precision/scripts/new_llm/venv_cerebras_pt

echo "Script execution completed at $(date). Virtual environment removed."