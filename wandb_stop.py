import wandb
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    args = parser.parse_args()
    tuner = wandb.controller(sweep_id_or_config = args.sweep_id, project = args.project, entity='yanyiphei')
    tuner._sweep_object_read_from_backend()
    has_killed = False
    while not has_killed:
        tuner._sweep_object_read_from_backend()
        a = tuner._stopping()
        print(f"Number of current runs fetched: {len(tuner._sweep_runs)}")
        if len(a) > 0:
            print("KILLING")
            tuner.stop_runs(a)
            has_killed = True
        time.sleep(1)

    print("PROGRAM TERMINATED")