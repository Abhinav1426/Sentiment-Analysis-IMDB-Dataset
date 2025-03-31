import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import word_tokenize
from wordcloud import WordCloud
import collections
import json

class Data_Visualizer:
    def __init__(self, df):
        """
        Initialize the Data_Visualizer class with a DataFrame.
        Args:
            df: DataFrame containing the dataset.
        """
        self.df = df

    @staticmethod
    def generate_wordcloud(df, sentiment, title):
        """
        Generate and display a word cloud for a specific sentiment.
        Args:
            df: DataFrame containing the dataset.
            sentiment: Sentiment value (e.g., 1 for positive, 0 for negative).
            title: Title for the word cloud plot.
        """
        sns.set(style="white", font_scale=1.2)
        plt.figure(figsize=(20, 20))
        # Generate word cloud from reviews with the specified sentiment
        wc = WordCloud(max_words=2000, width=1600, height=800).generate(" ".join(df[df.sentiment == sentiment].review))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def wordcloud_positive(self):
        """
        Generate a word cloud for positive reviews.
        """
        self.generate_wordcloud(self.df, 1, "Positive Review Text")

    def wordcloud_negative(self):
        """
        Generate a word cloud for negative reviews.
        """
        self.generate_wordcloud(self.df, 0, "Negative Review Text")

    def sentiment_distribution(self):
        """
        Plot the distribution of positive and negative reviews.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x='sentiment', data=self.df, palette='viridis')
        plt.title('Distribution of Positive and Negative Reviews')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'])
        plt.show()

    def unigram_analysis(self, top_n=20):
        """
        Perform unigram analysis to identify the most common words in reviews.
        Args:
            top_n: Number of top unigrams to display.
        """
        # Combine all reviews into a single string and tokenize
        all_words = ' '.join(self.df['review']).lower()
        tokens = word_tokenize(all_words)
        # Count the frequency of each unigram
        unigram_counts = collections.Counter(tokens)
        common_unigrams = unigram_counts.most_common(top_n)

        # Extract unigrams and their counts
        unigrams, counts = zip(*common_unigrams)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(counts), y=list(unigrams), palette='Paired')
        plt.title(f'Top {top_n} Common Words in Reviews')
        plt.xlabel('Frequency')
        plt.ylabel('Unigrams')
        plt.show()

    @staticmethod
    def plot_metrics(model):
        """
        Plot performance metrics (accuracy, precision, recall, F1 score) for a given model.
        Args:
            model: Model name ('bert' or 'lstm').
        """
        # Read metrics from the corresponding JSON file
        metrics_file = 'models/bert/metrics.json' if model == 'bert' else 'models/lstm/metrics.json'
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Prepare data for the bar plot
        metrics_values = [
            metrics['accuracy'] * 100,
            metrics['precision'] * 100,
            metrics['recall'] * 100,
            metrics['f1_score'] * 100
        ]
        metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_labels, metrics_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f1c40f'])

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}%',
                     ha='center', va='bottom')

        plt.title(f'{"BERT" if model == "bert" else "LSTM"} Model Performance Metrics')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)  # Set y-axis limit to 100%
        plt.show()

    @staticmethod
    def compare_models():
        """
        Compare the performance of BERT and LSTM models using a radar chart.
        """
        try:
            # Read metrics from both JSON files
            with open('models/bert/metrics.json', 'r') as f:
                bert_metrics = json.load(f)
            with open('models/lstm/metrics.json', 'r') as f:
                lstm_metrics = json.load(f)

            # Prepare data for the radar chart
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

            bert_values = [
                bert_metrics['accuracy'] * 100,
                bert_metrics['precision'] * 100,
                bert_metrics['recall'] * 100,
                bert_metrics['f1_score'] * 100
            ]

            lstm_values = [
                lstm_metrics['accuracy'] * 100,
                lstm_metrics['precision'] * 100,
                lstm_metrics['recall'] * 100,
                lstm_metrics['f1_score'] * 100
            ]

            # Repeat the first value to close the polygon
            bert_values += [bert_values[0]]
            lstm_values += [lstm_values[0]]
            metrics += [metrics[0]]

            # Compute angle for each metric
            angles = [n / float(len(metrics) - 1) * 2 * np.pi for n in range(len(metrics))]

            # Set up the radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

            # Plot BERT and LSTM data
            ax.plot(angles, bert_values, 'o-', linewidth=2, label='BERT', color='red')
            ax.fill(angles, bert_values, alpha=0.25, color='red')

            ax.plot(angles, lstm_values, 'o-', linewidth=2, label='LSTM', color='yellow')
            ax.fill(angles, lstm_values, alpha=0.25, color='yellow')

            # Add labels for each metric
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics[:-1])

            # Add value labels for each metric
            for angle, bert_value, lstm_value in zip(angles[:-1], bert_values[:-1], lstm_values[:-1]):
                ax.text(angle, bert_value + 1, f'{bert_value:.2f}%',
                        ha='center', va='bottom')
                ax.text(angle, lstm_value - 1, f'{lstm_value:.2f}%',
                        ha='center', va='top')

            # Set chart properties
            ax.set_ylim(80, 90)
            plt.title('BERT vs LSTM Model Performance Comparison')
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.show()

        except FileNotFoundError as e:
            print(f"Error: Could not find metrics files. Make sure they exist in the correct path.\n{str(e)}")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in metrics files.\n{str(e)}")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")