
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    )
    parser.add_argument("--quantity", type=int, default = 1)
    parser.add_argument("--region", type=str, default = "us-west-1")
    parser.add_argument("--instance_type", type=str, default = "gpu_1x_a10")
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.getenv("CLOUD_LAMBDA")
    url = "https://cloud.lambdalabs.com/api/v1/instances"

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.get(url, headers=headers, auth=HTTPBasicAuth(api_key, ''))

    # Check if the request was successful
    if response.status_code == 200:
        print('Success!')
    else:
        print('An error has occurred.')
        print(response.json().get('error', None))
        print(response.json().get('field_errors', None))
        raise Exception()

    instances = response.json()['data']
    print(f"Instances number: {len(instances)}")