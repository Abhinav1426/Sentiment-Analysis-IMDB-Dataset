import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

os.makedirs('models/lstm', exist_ok=True)

class LSTM_Model:
    def __init__(self, load_data_fn):
        self.load_data_fn = load_data_fn
        self.device = None
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        print("Initialized LSTM model class")

    def tokenize_data(self, texts, labels, max_length=128):
        print(f"Tokenizing {len(texts)} texts")
        input_ids = []
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = text.split()
            indices = [self.tokenizer.get(token, self.tokenizer.get('<UNK>', 0)) for token in tokens]
            if len(indices) < max_length:
                indices = indices + [self.tokenizer.get('<PAD>', 0)] * (max_length - len(indices))
            else:
                indices = indices[:max_length]
            input_ids.append(torch.tensor(indices))
        input_ids = torch.stack(input_ids)
        labels = torch.tensor(labels)
        print("Tokenization completed")
        return input_ids, labels

    def fine_tune_model(self, model, train_dataloader, val_dataloader, epochs=2):
        print(f"Starting fine-tuning for {epochs} epochs")
        best_val_accuracy = 0
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}/{epochs}")
            model.train()
            total_train_loss = 0

            for batch in tqdm(train_dataloader, desc="Training"):
                batch_input_ids = batch[0].to(self.device)
                batch_labels = batch[1].to(self.device)

                model.zero_grad()
                outputs = model(batch_input_ids)
                loss = criterion(outputs, batch_labels)
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss:.4f}")

            model.eval()
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    batch_input_ids = batch[0].to(self.device)
                    batch_labels = batch[1].to(self.device)

                    outputs = model(batch_input_ids)
                    predictions = torch.argmax(outputs, dim=1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_true_labels.extend(batch_labels.cpu().numpy())

            val_accuracy = accuracy_score(val_true_labels, val_predictions)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), 'models/lstm/lstm_model.h5')
                print(f"New best model saved with accuracy: {val_accuracy:.4f}")

        return model

    @staticmethod
    def create_dataloader(inputs, labels, batch_size, is_train=False):
        dataset = TensorDataset(inputs, labels)
        sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=2,
            drop_last=is_train
        )

    def load_model(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=2):
        print("Loading LSTM model...")
        class LSTMClassifier(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
                super(LSTMClassifier, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                embeds = self.embedding(x)
                lstm_out, _ = self.lstm(embeds)
                last_out = lstm_out[:, -1, :]
                logits = self.fc(last_out)
                return logits

        vocab_size = len(self.tokenizer) if self.tokenizer else vocab_size
        model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)
        self.model = model
        print("LSTM model loaded successfully")
        return model

    def evaluate(self, test_dataloader):
        print("Starting model evaluation")
        self.model.eval()
        test_predictions = []
        test_true_labels = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing"):
                batch_input_ids = batch[0].to(self.device)
                batch_labels = batch[1].to(self.device)

                outputs = self.model(batch_input_ids)
                predictions = torch.argmax(outputs, dim=1)
                test_predictions.extend(predictions.cpu().numpy())
                test_true_labels.extend(batch_labels.cpu().numpy())

        print("Evaluation completed")
        return test_predictions, test_true_labels

    @staticmethod
    def calculate_metrics(test_predictions, test_true_labels):
        print("Calculating metrics")
        accuracy = accuracy_score(test_true_labels, test_predictions)
        precision = precision_score(test_true_labels, test_predictions, average='binary')
        recall = recall_score(test_true_labels, test_predictions, average='binary')
        f1 = f1_score(test_true_labels, test_predictions, average='binary')

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
        print("Starting LSTM model training")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data_fn()
        print("Data splits loaded successfully")

        # Build a simple tokenizer (vocabulary) from training data
        self.tokenizer = {'<PAD>': 0, '<UNK>': 1}
        for text in X_train:
            for token in text.split():
                if token not in self.tokenizer:
                    self.tokenizer[token] = len(self.tokenizer)
        print("Tokenizer built")

        train_inputs, train_labels = self.tokenize_data(X_train, y_train)
        val_inputs, val_labels = self.tokenize_data(X_val, y_val)
        test_inputs, test_labels = self.tokenize_data(X_test, y_test)

        batch_size = 16
        train_dataloader = self.create_dataloader(train_inputs, train_labels, batch_size, is_train=True)
        val_dataloader = self.create_dataloader(val_inputs, val_labels, batch_size)
        test_dataloader = self.create_dataloader(test_inputs, test_labels, batch_size)
        print("DataLoaders created successfully")

        vocab_size = len(self.tokenizer)
        self.load_model(vocab_size)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model = self.fine_tune_model(self.model, train_dataloader, val_dataloader, epochs=3)
        self.model.load_state_dict(torch.load('models/lstm/lstm_model.h5'))

        test_predictions, test_true_labels = self.evaluate(test_dataloader)
        self.calculate_metrics(test_predictions, test_true_labels)

        print("Training completed successfully")