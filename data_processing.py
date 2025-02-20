import pandas as pd
from tqdm import tqdm

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

