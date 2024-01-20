import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.data)
    df_titanic = pd.read_csv(args.data)
    mlflow.log_metric("num_samples", df_titanic.shape[0])
    mlflow.log_metric("num_features", df_titanic.shape[1] - 1)
    df_titanic_train, df_titanic_test = train_test_split(df_titanic,
                                                         test_size=args.test_train_ratio, 
                                                         random_state=0)

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    df_titanic_train.to_csv(os.path.join(args.train_data, "titanic_train.csv"), index=False)
    df_titanic_test.to_csv(os.path.join(args.test_data, "titanic_test.csv"), index=False)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()