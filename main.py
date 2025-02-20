from data_processing import Data_Processer
from data_visualizer import Data_Visualizer
from text_processer import Text_Processer

if __name__ == "__main__":
    # Initialize text processor and data processor
    text_processor = Text_Processer()
    processor = Data_Processer('files/dataset.csv')

    # Load and display data
    processor.load_data()
    processor.display_basic_info()
    processor.convert_labels_binary()

    # Initialize data visualizer and generate visualizations
    data_visualizer = Data_Visualizer(processor.df)
    data_visualizer.sentiment_distribution()
    data_visualizer.wordcloud_positive()
    data_visualizer.wordcloud_negative()
    data_visualizer.unigram_analysis()

    # Preprocess text data
    processor.preprocess_text(text_processor)
