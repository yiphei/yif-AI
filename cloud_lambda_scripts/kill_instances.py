import argparse
import os

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("CLOUD_LAMBDA")
    url = "https://cloud.lambdalabs.com/api/v1/instances"

    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers, auth=HTTPBasicAuth(api_key, ""))

    # Check if the request was successful
    if response.status_code == 200:
        print("Success!")
    else:
        print("An error has occurred.")
        print(response.json().get("error", None))
        print(response.json().get("field_errors", None))
        raise Exception()

    instances = response.json()["data"]
    instance_ids = [instance["id"] for instance in instances if (args.name is None or instance.get("name", None) == args.name)]

    url = "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate"
    data = {
        "instance_ids": instance_ids,
    }

    response = requests.post(
        url, json=data, headers=headers, auth=HTTPBasicAuth(api_key, "")
    )
    if response.status_code == 200:
        print("Success!")
    else:
        print("An error has occurred.")
        print(response.json().get("error", None))
        print(response.json().get("field_errors", None))
        raise Exception()
