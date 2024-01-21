import os 
import pandas as pd


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