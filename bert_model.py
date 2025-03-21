import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import os


BERT_MODEL = 'bert-base-uncased'
os.makedirs('models/bert', exist_ok=True)

class Bert_Model:
    def __init__(self, load_data_fn):
        self.load_data_fn = load_data_fn
        self.device = None
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        print("Initialized BERT model class")
    def tokenize_data(self, texts, labels, max_length=128):
        print(f"Tokenizing {len(texts)} texts")
        input_ids = []
        attention_masks = []

        for text in tqdm(texts, desc="Tokenizing"):
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        print("Tokenization completed")
        return input_ids, attention_masks, labels
    def fine_tune_model(self, model, train_dataloader, val_dataloader, epochs=2):
        print(f"Starting fine-tuning for {epochs} epochs")
        best_val_accuracy = 0
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}/{epochs}")

            # Training
            model.train()
            total_train_loss = 0

            for batch in tqdm(train_dataloader, desc="Training"):
                batch_input_ids = batch[0].to(self.device)
                batch_attention_mask = batch[1].to(self.device)
                batch_labels = batch[2].to(self.device)

                model.zero_grad()
                outputs = model(
                    batch_input_ids,
                    token_type_ids=None,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )

                loss = outputs.loss
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss:.4f}")

            # Validation
            model.eval()
            val_predictions = []
            val_true_labels = []

            for batch in tqdm(val_dataloader, desc="Validation"):
                batch_input_ids = batch[0].to(self.device)
                batch_attention_mask = batch[1].to(self.device)
                batch_labels = batch[2].to(self.device)

                with torch.no_grad():
                    outputs = model(
                        batch_input_ids,
                        token_type_ids=None,
                        attention_mask=batch_attention_mask
                    )

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(batch_labels.cpu().numpy())

            val_accuracy = accuracy_score(val_true_labels, val_predictions)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), 'models/bert/bert_model.pt')
                print(f"New best model saved with accuracy: {val_accuracy:.4f}")
        return model
    @staticmethod
    def create_dataloader(inputs, masks, labels, batch_size, is_train=False):
        dataset = TensorDataset(inputs, masks, labels)
        sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=2,
            drop_last=is_train
        )
    def load_model(self):
        print("Loading BERT model...")
        model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        self.model = model
        print("BERT model loaded successfully")
        return model
    def evaluate(self, test_dataloader):
        print("Starting model evaluation")
        self.model.eval()
        test_predictions = []
        test_true_labels = []

        for batch in tqdm(test_dataloader, desc="Testing"):
            batch_input_ids = batch[0].to(self.device)
            batch_attention_mask = batch[1].to(self.device)
            batch_labels = batch[2].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    batch_input_ids,
                    token_type_ids=None,
                    attention_mask=batch_attention_mask
                )

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            test_predictions.extend(predictions.cpu().numpy())
            test_true_labels.extend(batch_labels.cpu().numpy())

        print("Evaluation completed")
        return test_predictions, test_true_labels
    @staticmethod
    def calculate_metrics(test_predictions, test_true_labels):
        print("Calculating metrics")
        accuracy = accuracy_score(test_true_labels, test_predictions)
        precision = precision_score(test_true_labels, test_predictions)
        recall = recall_score(test_true_labels, test_predictions)
        f1 = f1_score(test_true_labels, test_predictions)

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }

        print(f"Test Metrics: {metrics}")
        with open('models/bert/metrics.json', 'w') as f:
            json.dump(metrics, f)
    def train_model(self):
        print("Starting BERT model training")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print("Loading data splits...")
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data_fn()  # Call the function
        print("Data splits loaded successfully")

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        print("BERT tokenizer loaded")

        train_inputs, train_masks, train_labels = self.tokenize_data(X_train, y_train)
        val_inputs, val_masks, val_labels = self.tokenize_data(X_val, y_val)
        test_inputs, test_masks, test_labels = self.tokenize_data(X_test, y_test)

        batch_size = 16
        train_dataloader = self.create_dataloader(train_inputs, train_masks, train_labels, batch_size, is_train=True)
        val_dataloader = self.create_dataloader(val_inputs, val_masks, val_labels, batch_size)
        test_dataloader = self.create_dataloader(test_inputs, test_masks, test_labels, batch_size)
        print("DataLoaders created successfully")

        self.load_model()
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

        self.model = self.fine_tune_model(self.model, train_dataloader, val_dataloader, epochs=3)
        self.model.load_state_dict(torch.load('models/bert/bert_model.pt'))

        self.tokenizer.save_pretrained('models/bert/tokenizer')
        print("Tokenizer saved successfully")

        test_predictions, test_true_labels = self.evaluate(test_dataloader)
        self.calculate_metrics(test_predictions, test_true_labels)

        print("Training completed successfully")