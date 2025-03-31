import json
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# Create the directory for saving the LSTM model if it doesn't exist
os.makedirs('models/lstm', exist_ok=True)

# Constants for tokenization and padding
MAX_NUM_WORDS = 25000  # Maximum number of words to keep in the tokenizer
MAX_SEQUENCE_LENGTH = 200  # Maximum length of sequences after padding

class LSTM_Model:
    def __init__(self, load_data_fn):
        """
        Initialize the LSTM_Model class.
        Args:
            load_data_fn: Function to load the dataset splits.
        """
        self.load_data_fn = load_data_fn
        self.tokenizer = None
        self.model = None

        print("Initialized LSTM model class")

    def load_model(self):
        """
        Build and compile the LSTM model.
        """
        print("Building LSTM model...")
        embedding_dim = 128  # Dimension of the embedding layer
        vocab_size = min(MAX_NUM_WORDS, len(self.tokenizer.word_index) + 1)  # Vocabulary size

        # Define the LSTM model architecture
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=MAX_SEQUENCE_LENGTH))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))  # First LSTM layer
        model.add(Bidirectional(LSTM(32)))  # Second LSTM layer
        model.add(Dense(64, activation='relu'))  # Fully connected layer
        model.add(Dropout(0.5))  # Dropout for regularization
        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

        # Compile the model with loss, optimizer, and metrics
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        self.model = model

    def tokenize_data(self, X_train):
        """
        Tokenize the training data and save the tokenizer.
        Args:
            X_train: List of training text data.
        """
        print("Tokenizing text...")
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')  # Initialize tokenizer
        tokenizer.fit_on_texts(X_train)  # Fit tokenizer on training data

        # Save the tokenizer to a JSON file
        with open('models/lstm/tokenizer.json', 'w') as f:
            f.write(tokenizer.to_json())
        self.tokenizer = tokenizer

    @staticmethod
    def calculate_metrics(y_test, y_pred):
        """
        Calculate and save evaluation metrics.
        Args:
            y_test: True labels.
            y_pred: Predicted labels.
        """
        # Calculate accuracy, precision, recall, and F1 score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Print the metrics
        print(f"LSTM Test Accuracy: {accuracy:.4f}")
        print(f"LSTM Test Precision: {precision:.4f}")
        print(f"LSTM Test Recall: {recall:.4f}")
        print(f"LSTM Test F1 Score: {f1:.4f}")

        # Save the metrics to a JSON file
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        print(f"Test Metrics: {metrics}")
        with open('models/lstm/metrics.json', 'w') as f:
            json.dump(metrics, f)

    def train_model(self):
        """
        Train the LSTM model on the dataset.
        """
        print("Loading data splits...")

        # Load the training, validation, and test data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data_fn()

        # Tokenize the training data
        self.tokenize_data(X_train)

        # Convert text data to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq = self.tokenizer.texts_to_sequences(X_val)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)

        # Pad the sequences to ensure uniform length
        X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

        # Build the LSTM model
        self.load_model()

        # Train the model on the training data
        print("Training LSTM model...")
        history = self.model.fit(
            X_train_pad, y_train,
            validation_data=(X_val_pad, y_val),
            epochs=5,  # Number of training epochs
            batch_size=64,  # Batch size for training
            verbose=1  # Verbosity level
        )

        # Evaluate the model on the test data
        print("Evaluating LSTM model...")
        y_pred_proba = self.model.predict(X_test_pad)  # Predict probabilities
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions

        # Calculate and save evaluation metrics
        self.calculate_metrics(y_test, y_pred)

        # Save the trained model
        self.model.save('models/lstm/lstm_model.h5')

        print("LSTM model and metrics saved to models/lstm/")