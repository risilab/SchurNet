#!/bin/bash

MASTER_ADDR=localhost

# Function to check if a port is available
is_port_available() {
  local port=$1
  (echo >/dev/tcp/localhost/$port) &>/dev/null
  if [ $? -eq 0 ]; then
    return 1  # Port is in use
  else
    return 0  # Port is available
  fi
}

# Generate a random port number within a specified range (e.g., 12000 to 13000)
generate_random_port() {
  while : ; do
    local port=$((RANDOM % 1000 + 12000))
    if is_port_available $port; then
      echo $port
      break
    fi
  done
}
echo "starting"
echo "num gpu is $NUM_GPUS"
# Set the master port to a random available port
MASTER_PORT=$(generate_random_port)
CONFIG_NAME=$1
echo "config name is $CONFIG_NAME"



# Set NCCL timeout (e.g., 30 minutes)
export NCCL_TIMEOUT=1800  # 30 minutes (1800 seconds)
export TORCH_NCCL_BLOCKING_WAIT=1

# Determine the number of nodes and processes
n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: $n_gpus"
nnodes=1
nproc_per_node=$n_gpus
WORLD_SIZE=$n_gpus

# Build the command
command="torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT main_batch_ddp.py"



if [ -n "$CONFIG_NAME" ]; then
    command+=" --config-name=$CONFIG_NAME"
else
    echo "No config name provided"
fi


# Execute the command
echo "Executing command: $command"
eval $command
