import re
import nltk
import string

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

class Text_Processer:
    """
    A class for text preprocessing, including cleaning, stemming, and lemmatization.
    """

    @staticmethod
    def remove_html_tags(text):
        """
        Remove HTML tags from the text.
        Args:
            text: Input text containing HTML tags.
        Returns:
            Cleaned text without HTML tags.
        """
        pattern = re.compile('<.*?>')
        cleaned_text = pattern.sub(r'', text)
        return cleaned_text

    @staticmethod
    def remove_between_brackets(text):
        """
        Remove text enclosed in square brackets.
        Args:
            text: Input text containing square brackets.
        Returns:
            Cleaned text without content inside square brackets.
        """
        cleaned_text = re.sub('\[[^]]*\]', '', text)
        return cleaned_text

    @staticmethod
    def remove_url(text):
        """
        Remove URLs from the text.
        Args:
            text: Input text containing URLs.
        Returns:
            Cleaned text without URLs.
        """
        cleaned_text = re.sub(r'https?://\S+|www\.\S+', '', text)
        return cleaned_text

    @staticmethod
    def remove_special_characters(text):
        """
        Remove special characters from the text.
        Args:
            text: Input text containing special characters.
        Returns:
            Cleaned text with only alphanumeric characters and spaces.
        """
        cleaned_text = re.sub(r'[^a-zA-z0-9\s]', '', text)
        return cleaned_text

    @staticmethod
    def basic_stemmer(text):
        """
        Apply stemming to the text using PorterStemmer.
        Args:
            text: Input text to be stemmed.
        Returns:
            Stemmed text.
        """
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text

    @staticmethod
    def final_cleaning(text):
        """
        Remove stopwords and punctuation, and apply lemmatization.
        Args:
            text: Input text to be cleaned.
        Returns:
            Fully cleaned and lemmatized text.
        """
        stop = set(stopwords.words('english'))  # Load English stopwords
        punctuation = list(string.punctuation)  # List of punctuation characters
        stop.update(punctuation)  # Add punctuation to stopwords
        lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
        final_text = []

        # Tokenize the text into words
        for word in word_tokenize(text):
            # Check if the word is not a stopword and is alphabetic
            if word not in stop and word.isalpha():
                # Lemmatize the word
                final_text.append(lemmatizer.lemmatize(word))
        # Join the cleaned words back into a single string
        return " ".join(final_text)

    def clean_text(self, text):
        """
        Apply all preprocessing steps to clean the text.
        Args:
            text: Input raw text.
        Returns:
            Fully cleaned text.
        """
        text = text.lower()  # Convert text to lowercase
        text = self.remove_html_tags(text)  # Remove HTML tags
        text = self.remove_url(text)  # Remove URLs
        text = self.remove_between_brackets(text)  # Remove text between brackets
        text = self.remove_special_characters(text)  # Remove special characters
        text = self.basic_stemmer(text)  # Apply stemming
        text = self.final_cleaning(text)  # Remove stopwords and lemmatize
        return text