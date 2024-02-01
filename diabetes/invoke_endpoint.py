from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os
from dotenv import load_dotenv
import urllib.request
import json
import ssl
import pandas as pd
from random import randint
from time import sleep
import argparse


load_dotenv("../python.env")
ENDPOINT_API_KEY = os.getenv("ENDPOINT_API_KEY")
SUBSCRIPTION_ID = "e5615bfe-b43b-41ce-bccb-b78867c2ce63"
RESOURCE_GROUP = "rg-dp100-demo-001"
WORKSPACE_NAME = "mlw-dp100-demo"

ml_client = MLClient(DefaultAzureCredential(), 
                     subscription_id=SUBSCRIPTION_ID, 
                     resource_group_name=RESOURCE_GROUP,
                     workspace_name=WORKSPACE_NAME)


def get_prediction(input_data: list, api_key:str):
    data =  {
    "input_data": input_data,
    "params": {}
    }

    body = str.encode(json.dumps(data))
    url = 'https://endpoint-diabetes-prediction.japaneast.inference.ml.azure.com/score'
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'blue' }
    req = urllib.request.Request(url, body, headers)
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_batch", dest='num_batch', help='number of batches to test the endpoint',
                        type=int, default=5)
    args = parser.parse_args()
    return args

def main(args):
    df = pd.read_csv('./data/diabetes.csv')
    df = df.drop(columns=['Outcome'], index=1)
    for i in range(args.num_batch):
        sample_data = df.sample(randint(20,50)).values.tolist()
        print(f"Iteration: {i+1}")
        output = get_prediction(sample_data, ENDPOINT_API_KEY)
        print(output)
        sleep(randint(1,3))


if __name__ == '__main__':
    print("\n\n")
    print("*" * 60)

    args = parse_args()
    main(args)

    print("*" * 60)
    print("\n\n")