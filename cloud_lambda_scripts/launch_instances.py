import argparse
import os
import time

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--region", type=str, default="us-west-1")
    parser.add_argument("--instance_type", type=str, default="gpu_1x_a10")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("CLOUD_LAMBDA")
    url = "https://cloud.lambdalabs.com/api/v1/instance-operations/launch"
    data = {
        "region_name": args.region,
        "instance_type_name": args.instance_type,
        "ssh_key_names": ["lambda"],
        "quantity": 1,
    }

    headers = {"Content-Type": "application/json"}

    for i in range(args.quantity):
        response = requests.post(
            url, json=data, headers=headers, auth=HTTPBasicAuth(api_key, "")
        )

        # Check if the request was successful
        if response.status_code == 200:
            print("Success!")
        else:
            print("An error has occurred.")
            print(response.json().get("error", None))
            print(response.json().get("field_errors", None))
            raise Exception()

        print(response.text)

        if (i + 1) % 5 == 0:
            time.sleep(80)
