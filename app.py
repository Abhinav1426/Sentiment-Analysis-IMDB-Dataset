from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# Initialize Flask application
app = Flask(__name__)

# Load LSTM model and tokenizer
print("Loading LSTM model and tokenizer...")
lstm_model = tf.keras.models.load_model('models/lstm/lstm_model.h5')  # Load the trained LSTM model

with open('models/lstm/tokenizer.json', 'r') as f:
    lstm_tokenizer = tokenizer_from_json(f.read())  # Load the tokenizer for LSTM

lstm_max_length = 200  # Maximum sequence length for LSTM input

with open('models/lstm/metrics.json', 'r') as f:
    lstm_metrics = json.load(f)  # Load LSTM model metrics

# Load BERT model and tokenizer
print("Loading BERT model and tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU
bert_tokenizer = BertTokenizer.from_pretrained('models/bert/tokenizer')  # Load BERT tokenizer
bert_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)  # Load BERT model for sequence classification
bert_model.load_state_dict(torch.load('models/bert/bert_model.pt', map_location=device))  # Load trained BERT model weights
bert_model.to(device)  # Move BERT model to the appropriate device
bert_model.eval()  # Set BERT model to evaluation mode

with open('models/bert/metrics.json', 'r') as f:
    bert_metrics = json.load(f)  # Load BERT model metrics

print("Models loaded successfully!")

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from the request
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided. Please send a JSON with a "text" field.'}), 400

    text = data['text']

    # LSTM prediction
    lstm_seq = lstm_tokenizer.texts_to_sequences([text])  # Tokenize input text for LSTM
    lstm_padded = pad_sequences(lstm_seq, maxlen=lstm_max_length, padding='post', truncating='post')  # Pad sequences
    lstm_prediction = float(lstm_model.predict(lstm_padded)[0][0])  # Get LSTM prediction score
    lstm_sentiment = 'positive' if lstm_prediction >= 0.5 else 'negative'  # Determine sentiment
    lstm_confidence = lstm_prediction if lstm_prediction >= 0.5 else 1 - lstm_prediction  # Calculate confidence

    # BERT prediction
    encoded_text = bert_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )  # Tokenize and encode input text for BERT

    input_ids = encoded_text['input_ids'].to(device)  # Move input IDs to the appropriate device
    attention_mask = encoded_text['attention_mask'].to(device)  # Move attention mask to the appropriate device

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)  # Get BERT model outputs

    logits = outputs.logits  # Extract logits from BERT outputs
    probabilities = torch.nn.functional.softmax(logits, dim=1)  # Calculate probabilities
    bert_prediction = float(probabilities[0][1].item())  # Probability of positive class
    bert_sentiment = 'positive' if bert_prediction >= 0.5 else 'negative'  # Determine sentiment
    bert_confidence = bert_prediction if bert_prediction >= 0.5 else 1 - bert_prediction  # Calculate confidence

    # Ensemble prediction (average of both models)
    ensemble_score = (lstm_prediction + bert_prediction) / 2  # Average scores from both models
    ensemble_sentiment = 'positive' if ensemble_score >= 0.5 else 'negative'  # Determine sentiment
    ensemble_confidence = ensemble_score if ensemble_score >= 0.5 else 1 - ensemble_score  # Calculate confidence

    # Model comparison
    better_model = 'LSTM' if lstm_metrics['accuracy'] > bert_metrics['accuracy'] else 'BERT'  # Compare model accuracies
    if lstm_metrics['accuracy'] == bert_metrics['accuracy']:
        better_model = 'Both equal'  # Handle case where accuracies are equal

    # Models agreement
    models_agree = lstm_sentiment == bert_sentiment  # Check if both models agree on sentiment

    # Prepare response
    response = {
        'input_text': text,
        'lstm_prediction': {
            'sentiment': lstm_sentiment,
            'confidence': float(lstm_confidence),
            'raw_score': float(lstm_prediction),
            'metrics': lstm_metrics
        },
        'bert_prediction': {
            'sentiment': bert_sentiment,
            'confidence': float(bert_confidence),
            'raw_score': float(bert_prediction),
            'metrics': bert_metrics
        },
        'ensemble_prediction': {
            'sentiment': ensemble_sentiment,
            'confidence': float(ensemble_confidence),
            'raw_score': float(ensemble_score)
        },
        'comparison': {
            'better_accuracy_model': better_model,
            'accuracy_difference': abs(lstm_metrics['accuracy'] - bert_metrics['accuracy']),
            'models_agree': models_agree,
            'higher_confidence_model': 'LSTM' if lstm_confidence > bert_confidence else 'BERT' if bert_confidence > lstm_confidence else 'Both equal',
            'confidence_difference': abs(float(lstm_confidence - bert_confidence))
        }
    }

    return jsonify(response)  # Return the response as JSON

# Define the health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models': {
            'lstm': {
                'loaded': True,
                'accuracy': lstm_metrics['accuracy']
            },
            'bert': {
                'loaded': True,
                'accuracy': bert_metrics['accuracy']
            }
        }
    })  # Return the health status of the application and models

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Start the Flask server