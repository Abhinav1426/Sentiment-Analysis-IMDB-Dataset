# train_lstm.py
import json
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

os.makedirs('models/lstm', exist_ok=True)
# Tokenization and padding
MAX_NUM_WORDS = 25000
MAX_SEQUENCE_LENGTH = 200
class LSTM_Model:
    def __init__(self, load_data_fn):
        self.load_data_fn = load_data_fn
        self.tokenizer = None
        self.model=None

        print("Initialized LSTM model class")

    def load_model(self):
        # Build the LSTM model
        print("Building LSTM model...")
        embedding_dim = 128
        vocab_size = min(MAX_NUM_WORDS, len(self.tokenizer.word_index) + 1)

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=MAX_SEQUENCE_LENGTH))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        self.model = model
    def tokenize_data(self,X_train):
        print("Tokenizing text...")
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)

        # Save the tokenizer
        with open('models/lstm/tokenizer.json', 'w') as f:
            f.write(tokenizer.to_json())
        self.tokenizer = tokenizer
    @staticmethod
    def calculate_metrics(y_test, y_pred):

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"LSTM Test Accuracy: {accuracy:.4f}")
        print(f"LSTM Test Precision: {precision:.4f}")
        print(f"LSTM Test Recall: {recall:.4f}")
        print(f"LSTM Test F1 Score: {f1:.4f}")

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
        # Load the data splits
        print("Loading data splits...")

        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data_fn()
        self.tokenize_data(X_train)
        # Convert text to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq = self.tokenizer.texts_to_sequences(X_val)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)

        # Pad sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

        self.load_model()

        # Train the model
        print("Training LSTM model...")
        history = self.model.fit(
            X_train_pad, y_train,
            validation_data=(X_val_pad, y_val),
            epochs=5,
            batch_size=64,
            verbose=1
        )

        # Evaluate the model
        print("Evaluating LSTM model...")
        y_pred_proba = self.model.predict(X_test_pad)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        self.calculate_metrics(y_test, y_pred)

        # Save the model
        self.model.save('models/lstm/lstm_model.h5')

        print("LSTM model and metrics saved to models/lstm/")
