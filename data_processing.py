import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

class Data_Processer:
    def __init__(self, file_path):
        """
        Initialize the Data_Processer class.
        Args:
            file_path: Path to the dataset file.
        """
        self.file_path = file_path
        self.df = None  # DataFrame to hold the dataset

    def load_data(self):
        """
        Load the dataset from the specified file path.
        """
        try:
            self.df = pd.read_csv(self.file_path)  # Read the CSV file into a DataFrame
            print("Data loaded successfully!")
        except Exception as e:
            print("Error loading data:", e)

    def convert_labels_binary(self):
        """
        Convert sentiment labels to binary format.
        'positive' -> 1, 'negative' -> 0.
        """
        self.df.replace({"positive": 1, "negative": 0}, inplace=True)
        print("Labels converted to binary.")

    def display_basic_info(self):
        """
        Display basic information about the dataset, such as:
        - First few rows
        - Dataset structure
        - Summary statistics
        - Missing values
        """
        if self.df is not None:
            print("\nFirst few rows of the dataset:")
            print(self.df.head())  # Display the first few rows
            print("\nDataset info:")
            print(self.df.info())  # Display dataset structure
            print("\nDescribe the dataset:")
            print(self.df.describe())  # Display summary statistics
            print("\nChecking for the missing values:")
            print(self.df.isna().sum())  # Check for missing values
        else:
            print("Data not loaded. Please call load_data() first.")

    def preprocess_text(self, text_processor):
        """
        Preprocess the text data in the 'review' column using a text processor.
        Args:
            text_processor: An instance of a text processing class with a clean_text method.
        """
        tqdm.pandas(desc="Processing reviews")  # Add a progress bar for processing
        self.df['review'] = self.df['review'].progress_apply(text_processor.clean_text)  # Clean text data
        print("Reviews preprocessed.")

    def data_split_and_save(self):
        """
        Split the dataset into training, validation, and test sets.
        Save the splits to a .npz file for later use.
        """
        X = self.df['review'].values  # Extract review text
        y = self.df['sentiment'].values  # Extract sentiment labels

        # Split data into training and temporary sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        # Further split the temporary set into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Save the splits to a .npz file
        np.savez('files/data_splits.npz', X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val,
                 X_test=X_test, y_test=y_test)

    @staticmethod
    def data_loader():
        """
        Load the pre-saved dataset splits from a .npz file.
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test: Dataset splits.
        """
        data = np.load('files/data_splits.npz', allow_pickle=True)  # Load the .npz file
        X_train, y_train = data['X_train'], data['y_train']  # Training data
        X_val, y_val = data['X_val'], data['y_val']  # Validation data
        X_test, y_test = data['X_test'], data['y_test']  # Test data
        return X_train, y_train, X_val, y_val, X_test, y_test