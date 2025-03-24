from bert_model import Bert_Model
from data_processing import Data_Processer
from data_visualizer import Data_Visualizer
from lstm_model import LSTM_Model
from text_processer import Text_Processer

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Initialize text processor and data processor
    text_processor = Text_Processer()
    # dataset link : https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data
    processor = Data_Processer('files/dataset.csv')

    # Load and display data
    processor.load_data()
    processor.display_basic_info()
    processor.convert_labels_binary()

    # Preprocess text data
    processor.preprocess_text(text_processor)
    processor.data_split_and_save()

    # Initialize data visualizer and generate visualizations
    data_visualizer = Data_Visualizer(processor.df)
    data_visualizer.sentiment_distribution()  # Visualize sentiment distribution
    data_visualizer.wordcloud_positive()  # Generate word cloud for positive reviews
    data_visualizer.wordcloud_negative()  # Generate word cloud for negative reviews
    data_visualizer.unigram_analysis()  # Perform unigram analysis

    # Initialize and train BERT model
    bert_model = Bert_Model(processor.data_loader)
    bert_model.train_model()
    data_visualizer.plot_metrics('bert')
    # Initialize and train LSTM model
    lstm_model = LSTM_Model(processor.data_loader)
    lstm_model.train_model()
    data_visualizer.plot_metrics('lstm')
    data_visualizer.compare_models()



