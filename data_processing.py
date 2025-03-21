import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np


class Data_Processer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print("Data loaded successfully!")
        except Exception as e:
            print("Error loading data:", e)

    def convert_labels_binary(self):
        # Convert 'positive' to 1 and 'negative' to 0 in the DataFrame
        self.df.replace({"positive": 1, "negative": 0}, inplace=True)
        print("Labels converted to binary.")

    def display_basic_info(self):
        if self.df is not None:
            print("\nFirst few rows of the dataset:")
            print(self.df.head())
            print("\nDataset info:")
            print(self.df.info())
            print("\nDescribe the dataset:")
            print(self.df.describe())
            print("\nChecking for the missing values:")
            print(self.df.isna().sum())
        else:
            print("Data not loaded. Please call load_data() first.")

    def preprocess_text(self,text_processor):
        # Apply text cleaning to the 'review' column in the DataFrame
        tqdm.pandas(desc="Processing reviews")
        self.df['review'] = self.df['review'].progress_apply(text_processor.clean_text)
        print("Reviews preprocessed.")

    def data_split_and_save(self):
        X = self.df['review'].values
        y = self.df['sentiment'].values

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        np.savez('files/data_splits.npz', X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val,
                 X_test=X_test, y_test=y_test)

    @staticmethod
    def data_loader():
        data = np.load('files/data_splits.npz', allow_pickle=True)
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        return X_train, y_train , X_val, y_val , X_test, y_test

