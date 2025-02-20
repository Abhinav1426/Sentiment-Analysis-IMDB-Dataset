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

    @staticmethod
    def remove_html_tags(text):
        # Remove HTML tags from the text
        pattern = re.compile('<.*?>')
        cleaned_text = pattern.sub(r'', text)
        return cleaned_text

    @staticmethod
    def remove_between_brackets(text):
        # Remove text between brackets
        cleaned_text = re.sub('\[[^]]*\]', '', text)
        return cleaned_text

    @staticmethod
    def remove_url(text):
        # Remove URLs from the text
        cleaned_text = re.sub(r'https?://\S+|www\.\S+', '', text)
        return cleaned_text

    @staticmethod
    def remove_special_characters(text):
        cleaned_text = re.sub(r'[^a-zA-z0-9\s]', '', text)
        return cleaned_text

    @staticmethod
    # Stemming the text
    def basic_stemmer(text):
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text

    @staticmethod
    def final_cleaning(text):
        # Remove stopwords and punctuation from the text
        stop = set(stopwords.words('english'))
        punctuation = list(string.punctuation)
        stop.update(punctuation)
        # Initialize the WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
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

        # Clean the text by applying all preprocessing steps
        text = text.lower()
        text = self.remove_html_tags(text)
        text = self.remove_url(text)
        text = self.remove_between_brackets(text)
        text = self.remove_special_characters(text)
        text = self.basic_stemmer(text)
        text = self.final_cleaning(text)
        return text