from bert_model import Bert_Model
from data_processing import Data_Processer
from data_visualizer import Data_Visualizer
from lstm_model import LSTM_Model
from text_processer import Text_Processer

import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Initialize text processor and data processor
    text_processor = Text_Processer()
    # Dataset link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data
    processor = Data_Processer('files/dataset.csv')

    # Load and display basic information about the dataset
    processor.load_data()  # Load the dataset from the specified file
    processor.display_basic_info()  # Display basic statistics and structure of the dataset
    processor.convert_labels_binary()  # Convert sentiment labels to binary format (e.g., positive/negative)

    # Preprocess text data
    processor.preprocess_text(text_processor)  # Apply text cleaning and preprocessing steps
    processor.data_split_and_save()  # Split the data into training, validation, and test sets and save them

    # Initialize data visualizer and generate visualizations
    data_visualizer = Data_Visualizer(processor.df)
    data_visualizer.sentiment_distribution()  # Visualize the distribution of sentiments in the dataset
    data_visualizer.wordcloud_positive()  # Generate a word cloud for positive reviews
    data_visualizer.wordcloud_negative()  # Generate a word cloud for negative reviews
    data_visualizer.unigram_analysis()  # Perform unigram analysis to identify common words

    # Initialize and train the BERT model
    bert_model = Bert_Model(processor.data_loader)  # Pass the data loader to the BERT model
    bert_model.train_model()  # Train the BERT model on the dataset
    data_visualizer.plot_metrics('bert')  # Plot the performance metrics for the BERT model

    # Initialize and train the LSTM model
    lstm_model = LSTM_Model(processor.data_loader)  # Pass the data loader to the LSTM model
    lstm_model.train_model()  # Train the LSTM model on the dataset
    data_visualizer.plot_metrics('lstm')  # Plot the performance metrics for the LSTM model

    # Compare the performance of the BERT and LSTM models
    data_visualizer.compare_models()  # Generate a comparison visualization for both models