#!/bin/bash

# Log File Path
LOG_FILE=~/attention_layer_test.log

# Output error to Log
exec > >(tee -i $LOG_FILE) 2>&1

# Step 1: Clean up existing virtual environment if it exists
echo "Cleaning up any existing virtual environment..."
if [ -d "~/attention_layer_env" ]; then
  rm -rf ~/attention_layer_env
fi

# Step 2: Set up a new virtual environment
echo "Setting up a virtual environment..."
/opt/python3.8/bin/python3.8 -m venv ~/attention_layer_env
source ~/attention_layer_env/bin/activate

# Step 3: Upgrade pip and install required packages
pip install --upgrade pip
pip install cerebras_pytorch==2.3.1

pip install --verbose -r /home/kevienzzq/mixed_precision/R_2.3.1/modelzoo/requirements.txt
#pip install jsonschema
#pip install packaging


export PYTHONPATH=/home/kevienzzq/mixed_precision/R_2.3.1/modelzoo/src:$PYTHONPATH

# Step 4: Run the attention layer throughput test
echo "Running the attention layer throughput test..."
python /home/kevienzzq/mixed_precision/scripts/new_llm/test_attention.py  |& tee -a $LOG_FILE

# Step 5: Clean up the virtual environment
echo "Cleaning up the virtual environment..."
deactivate
rm -rf ~/attention_layer_env

echo "Script execution completed."
