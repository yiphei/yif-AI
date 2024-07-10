#!/bin/bash

if [ "$#" -lt 5 ] || [ "$#" -gt 7 ]; then
    echo "Usage: $0 <ADDRESS_FILE_PATH> <API_KEY> <AWS_ACCESS_KEY> <AWS_SECRET_KEY> <SYNC_S3> <SYNC_FILES> <GPU_PROCESSES>"
    exit 1
fi

# Assign the arguments to variables
ADDRESS_FILE=$1
API_KEY=$2
AWS_ACCESS_KEY=$3
AWS_SECRET_KEY=$4
SYNC_S3=${5:-false}
SYNC_FILES=${6:-true}
GPU_PROCESSES=${7:-1}

process_address() {
    address=$1
    api_key=$2
    aws_access_key=$3
    aws_secret_key=$4
    sync_s3=$5
    sync_files=$6
    gpu_processes=$7
    echo "Starting process for $address"

    if [[ "$(echo "$sync_files" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
        # Sync files
        echo "Syncing files to $address"
        git ls-files | rsync -avz --files-from=- -e "ssh -i ~/Downloads/lambda.pem -o StrictHostKeyChecking=no" ./ "$address":~/yif-AI/
    fi

    # SSH into the server, start a tmux session, and run the commands
    ssh -i ~/Downloads/lambda.pem -o StrictHostKeyChecking=no "$address" <<EOF
    tmux new-session -d -s mySession /bin/bash
    tmux send-keys -t mySession "cd yif-AI" C-m
    tmux send-keys -t mySession "pip install -r requirements.txt" C-m
    tmux send-keys -t mySession "pip install --upgrade pyOpenSSL cryptography boto3 botocore" C-m
    tmux send-keys -t mySession "pip install torchdata" C-m
    if [[ "$(echo "$sync_s3" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
        tmux send-keys -t mySession "mkdir datasets/openweb/" C-m
        tmux send-keys -t mySession "export AWS_ACCESS_KEY_ID=\"${aws_access_key}\"" C-m
        tmux send-keys -t mySession "export AWS_SECRET_ACCESS_KEY=\"${aws_secret_key}\"" C-m
        tmux send-keys -t mySession "aws s3 sync s3://dropout-transformer/datasets/openweb/ datasets/openweb/" C-m
    fi

    tmux send-keys -t mySession "tmux wait-for -S setup_done" C-m
    tmux wait-for setup_done

    export_cmd="export WANDB_API_KEY='${api_key}'"
    torch_cmd="torchrun --standalone --nproc_per_node=1 -m future_attention.training_script --config_file future_attention/train_configs/small.py --train datasets/wikipedia/ --platform_type LAMBDA --aws_access_key_id ${aws_access_key} --aws_secret_access_key ${aws_secret_key} --sweep_id csv2z2bv --sweep_count 1 --sync_profile_live True"

    if [ $gpu_processes -gt 1 ]; then
        for i in \$(seq 0 $((gpu_processes-1))); do
            session_name="mySession\$i"
            tmux new-session -d -s \$session_name /bin/bash
            tmux send-keys -t \$session_name "cd yif-AI" C-m
            tmux send-keys -t \$session_name "\$export_cmd" C-m
            tmux send-keys -t \$session_name "CUDA_VISIBLE_DEVICES=\$i \$torch_cmd" C-m
        done
    else
        tmux send-keys -t mySession "\$export_cmd" C-m
        tmux send-keys -t mySession "\$torch_cmd" C-m
    fi
    exit
EOF

    echo "Commands executed in tmux session for $address"
}

# Read each address from the file and process it in the background
while IFS=' ' read -r ssh_cmd address _ || [[ -n "$address" ]]; do
    if [[ -z "$address" ]]; then
        continue
    fi
    process_address "$address" "$API_KEY" "$AWS_ACCESS_KEY" "$AWS_SECRET_KEY" "$SYNC_S3" "$SYNC_FILES" "$GPU_PROCESSES" &
done < "$ADDRESS_FILE"

# Wait for all background processes to finish
wait
echo "All processes completed."
