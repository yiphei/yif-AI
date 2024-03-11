#!/bin/bash

ADDRESS_FILE="addresses.txt"

process_address() {
    address=$1
    echo "Starting process for $address"
    
    # Sync files
    echo "Syncing files to $address"
    git ls-files | rsync -avz --files-from=- -e "ssh -i ~/Downloads/lambda.pem -o StrictHostKeyChecking=no" ./ "$address":~/yif-AI/

    # SSH into the server, start a tmux session, and run the commands
    ssh -i ~/Downloads/lambda.pem -o StrictHostKeyChecking=no "$address" <<'EOF'
    tmux new-session -d -s mySession 'cd yif-AI && pip install -r requirements.txt'
    tmux send-keys -t mySession 'pip install --upgrade pyOpenSSL cryptography boto3 botocore' C-m
    tmux send-keys -t mySession 'torchrun --standalone --nproc_per_node=1 transformer_embed/training_script.py --config_file transformer_embed/train_configs/harry_potter_new.py --train datasets/full_harry_potter/ --platform_type LAMBDA --aws_access_key_id YOUR_ACCESS_KEY --aws_secret_access_key YOUR_SECRET_KEY' C-m
EOF

    echo "Commands executed in tmux session for $address"
}

# Read each address from the file and process it in the background
while IFS= read -r address
do
    process_address "$address" &
done < "$ADDRESS_FILE"

# Wait for all background processes to finish
wait
echo "All processes completed."
