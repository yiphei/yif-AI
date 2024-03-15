import wandb
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    args = parser.parse_args()
    tuner = wandb.controller(sweep_id_or_config = args.sweep_id, project = args.project, entity='yanyiphei')
    tuner.run(verbose=True)