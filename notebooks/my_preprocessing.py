from sklearn.utils import shuffle
import pandas as pd

def train_dev_test_split(df, column='location', pct_train=0.6, pct_dev=0.2, pct_test=0.2, random_state=42):
    """
    Splits up a dataset using stratified random sampling to ensure equal proportion of observations by column
    in each of the train, development, and test sets.
    
    Params: 
     - df (DataFrame): pandas dataframe you want to split into train
     - col (str): column you want to stratify on. Default value is 'location'
     - pct_train (float): percent of dataset you want in the training set reprsented as float between 0 and 1. Default value is 0.6
     - pct_dev (float): percent of the dataset you want in the development set reprsented as float between 0 and 1. Default value is 0.2
     - pct_test (float): percent of the dataset you want in the test set reprsented as float between 0 and 1. Default value is 0.2
    Returns:
    - tuple with train, dev, and test datasets as pandas dataframes
    """
    

    train = pd.DataFrame()
    dev = pd.DataFrame()
    test = pd.DataFrame()

    for value in df[column].unique():
        # Create a dataframe for that column value and shuffle it
        col_df = df[df[column] == value]
        col_df_shuffled = shuffle(col_df, random_state=random_state)

        # Create splits for train, dev, and test sets
        split_1 = int(pct_train * col_df.shape[0])
        split_2 = int((pct_train + pct_dev) * col_df.shape[0])

        # Split up shuffled dataframe (for each col)
        col_df_train = col_df_shuffled.iloc[:split_1]
        col_df_dev = col_df_shuffled.iloc[split_1:split_2]
        col_df_test = col_df_shuffled.iloc[split_2:]

        # Add on the selections for train, dev, and test
        train = pd.concat(objs=[train, col_df_train])
        dev = pd.concat(objs=[dev, col_df_dev])
        test = pd.concat(objs=[test, col_df_test])

    return train, dev, test