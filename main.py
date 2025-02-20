from data_processing import Data_Processer
from data_visualizer import Data_Visualizer
from text_processer import Text_Processer

import warnings
warnings.filterwarnings("ignore")

# google colab link
# https://colab.research.google.com/drive/1b9IW4hGieYXs2PPYwpsDbP9UkTViVeON?usp=sharing

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

    # Initialize data visualizer and generate visualizations
    data_visualizer = Data_Visualizer(processor.df)
    data_visualizer.sentiment_distribution()
    data_visualizer.wordcloud_positive()
    data_visualizer.wordcloud_negative()
    data_visualizer.unigram_analysis()


