import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import mlflow

# Helper Functions
def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

features = ['Pclass', 'Age', 'Fare', 'FamilySize', 'Sex_female', 'Sex_male']
# define a function to carry out the expected feature engineering steps for inference
def prepare_features(input_df: pd.DataFrame, features:list) -> pd.DataFrame:
    drop_list = ['PassengerId', 'Ticket', 'Name', 'Cabin', 'Embarked']
    drop_list = list(set(drop_list) & set(input_df.columns))
    input_df.drop(drop_list, axis=1, inplace=True)
    input_df['FamilySize'] = input_df['SibSp'] + input_df['Parch']
    input_df = pd.get_dummies(input_df, columns=['Sex'])
    if 'Sex_female' not in input_df.columns:
        input_df['Sex_female'] = False
    if 'Sex_male' not in input_df.columns:
        input_df['Sex_male'] = False
    input_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    input_df.fillna(input_df.median(), inplace=True)
    input_df = input_df[features]
    return input_df

# Start Logging
mlflow.start_run()
# enable autologging
mlflow.sklearn.autolog()
# create outputs folder if not exists
os.makedirs("./outputs", exist_ok=True)


def main():
    """Main function of the script."""
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--C", required=False, default=1, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()

    # Loading the training set and extracting the features and label
    df_train = pd.read_csv(select_first_file(args.train_data))
    y = df_train['Survived']
    df_train = prepare_features(df_train.drop(['Survived'], axis=1), features)
    X_train = df_train.values

    # Repeat for test set
    df_test = pd.read_csv(select_first_file(args.test_data))
    y_test = df_test['Survived']
    df_test = prepare_features(df_test.drop(['Survived'], axis=1), features)
    X_test = df_test.values  

    # Training and evaluating the model:
    print(f"Training with data of shape {X_train.shape}")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y)
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(sk_model=model,
                             registered_model_name=args.registered_model_name,
                             artifact_path=args.registered_model_name,
                             )

    # Saving the model to a file
    model_class_name = model.__class__.__name__
    mlflow.sklearn.save_model(sk_model=model, 
                              path=os.path.join(args.model, f"trained_{model_class_name}")
                              )
    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()