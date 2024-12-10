#!/bin/bash
# Log File Path (keeping your original path)
LOG_FILE=~/mixed_precision/output/test_attention.log
# Redirect output to log file
exec > >(tee -i $LOG_FILE) 2>&1
echo "Starting Test $(date)"
# Clean up existing virtual environment if it exists
# echo "Cleaning up any existing virtual environment..."
# if [ -d "~/mixed_precision/scripts/new_llm/venv_cerebras_pt" ]; then
#     rm -rf ~/mixed_precision/scripts/new_llm/venv_cerebras_pt
# fi

# Set up Python virtual environment (keeping your original path)
echo "Setting up Python virtual environment..."
/opt/python3.8/bin/python3.8 -m venv ~/venv_cerebras_pt
source ~/venv_cerebras_pt/bin/activate

# Install required packages
# echo "Installing required packages..."
# pip install --upgrade pip

# Clone or update ModelZoo repository (keeping your original path)
# mkdir -p ~/mixed_precision/R_2.3.0
# cd ~/mixed_precision/R_2.3.0
# if [ ! -d "modelzoo" ]; then
#     echo "Cloning ModelZoo repository..."
#     git clone https://github.com/Cerebras/modelzoo.git
#     cd modelzoo
#     git checkout Release_2.3.0
# else
#     echo "ModelZoo repository exists, updating..."
#     cd modelzoo
#     git pull
#     git checkout Release_2.3.0
# fi

# # Install ModelZoo requirements
# echo "Installing ModelZoo requirements..."
# pip install -r ~/mixed_precision/R_2.3.0/modelzoo/requirements.txt 

# Run training
echo "Starting test..."
cd /home/kevienzzq/dataflow-arch-test/scripts/new_llm
export PYTHONPATH=/home/$(whoami)/mixed_precision/R_2.3.0/modelzoo/src:$PYTHONPATH

python test_attention.py CSX \
    --job_labels name=test \
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