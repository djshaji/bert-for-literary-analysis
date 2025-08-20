import os
import random
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg

# 1. Load and preprocess public domain texts (e.g., Shakespeare, Austen, etc.)
def load_gutenberg_samples(authors=['austen', 'shakespeare'], n_per_author=100):
    texts, labels = [], []
    author_map = {
        'austen': [f for f in gutenberg.fileids() if 'austen' in f],
        'shakespeare': [f for f in gutenberg.fileids() if 'shakespeare' in f]
    }
    for label, author in enumerate(authors):
        files = author_map[author]
        for file in files:
            raw = gutenberg.raw(file)
            # Split into ~400 word samples for BERT
            chunks = [raw[i:i+1800] for i in range(0, len(raw), 1800)]
            for chunk in chunks[:n_per_author]:
                texts.append(chunk)
                labels.append(label)
    return texts, labels

# 2. Prepare Hugging Face dataset
def prepare_dataset(texts, labels):
    data = {'text': texts, 'label': labels}
    dataset = Dataset.from_dict(data)
    # Simple train/test split
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    return dataset

# 3. Tokenize
def tokenize_data(dataset, tokenizer):
    def tokenize_fn(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)
    return dataset.map(tokenize_fn, batched=True)

# 4. Fine-tune BERT for stylistic classification
def train_bert_classifier(dataset):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy='epoch',
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='no'
    )
    def compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        return {'accuracy': accuracy_score(pred.label_ids, preds)}
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer
 
# 5. Main script
if __name__ == "__main__":
    # Example: Stylistic classification (Austen vs Shakespeare)
    texts, labels = load_gutenberg_samples(authors=['austen', 'shakespeare'], n_per_author=50)
    dataset = prepare_dataset(texts, labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_dataset = tokenize_data(dataset, tokenizer)
    trainer = train_bert_classifier(tokenized_dataset)
    # Evaluate results
    preds = trainer.predict(tokenized_dataset['test'])
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(-1)
    print(classification_report(y_true, y_pred, target_names=['Austen', 'Shakespeare']))

    # Similar approach can be used for thematic and sentiment tasks