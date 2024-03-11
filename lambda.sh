#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <API_KEY> <AWS_ACCESS_KEY> <AWS_SECRET_KEY>"
    exit 1
fi

# Assign the arguments to variables
API_KEY=$1
AWS_ACCESS_KEY=$2
AWS_SECRET_KEY=$3

ADDRESS_FILE="addresses.txt"

process_address() {
    address=$1
    api_key=$2
    aws_access_key=$3
    aws_secret_key=$4
    echo "Starting process for $address"
    
    # Sync files
    echo "Syncing files to $address"
    git ls-files | rsync -avz --files-from=- -e "ssh -i ~/Downloads/lambda.pem -o StrictHostKeyChecking=no" ./ "$address":~/yif-AI/

    # SSH into the server, start a tmux session, and run the commands
    ssh -i ~/Downloads/lambda.pem -o StrictHostKeyChecking=no "$address" <<EOF
    tmux new-session -d -s mySession /bin/bash
    tmux send-keys -t mySession "cd yif-AI" C-m
    tmux send-keys -t mySession "pip install -r requirements.txt" C-m
    tmux send-keys -t mySession "pip install --upgrade pyOpenSSL cryptography boto3 botocore" C-m
    tmux send-keys -t mySession "export WANDB_API_KEY='${api_key}'" C-m
    tmux send-keys -t mySession "torchrun --standalone --nproc_per_node=1 -m transformer_dropout.training_script --config_file transformer_dropout/train_configs/harry_potter_baseline.py --train datasets/full_harry_potter/ --platform_type LAMBDA --aws_access_key_id ${aws_access_key} --aws_secret_access_key ${aws_secret_key}" C-m
    exit
EOF

    echo "Commands executed in tmux session for $address"
}

# Read each address from the file and process it in the background
while IFS= read -r address
do
    process_address "$address" "$API_KEY" "$AWS_ACCESS_KEY" "$AWS_SECRET_KEY" &
done < "$ADDRESS_FILE"

# Wait for all background processes to finish
wait
echo "All processes completed."
