"""
Thematic Motif Detection with BERT (Hugging Face Transformers)

This script demonstrates a simple framework for detecting the presence of a theme (e.g., 'love') in literary passages.
It uses keyword-based labeling for training data (weak supervision) and fine-tunes BERT for binary classification.

Requires: transformers, datasets, nltk, torch, scikit-learn
Run: python thematic_detection.py
"""
import nltk
from nltk.corpus import gutenberg
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report

nltk.download('gutenberg')

# Theme and keywords
theme = "love"
theme_keywords = ["love", "lover", "beloved", "sweetheart", "affection", "darling", "romance"]

def extract_samples_with_theme(n=200):
    texts, labels = [], []
    all_files = gutenberg.fileids()
    for fileid in all_files:
        text = gutenberg.raw(fileid)
        chunks = [text[i:i+1800] for i in range(0, len(text), 1800)]
        for chunk in chunks:
            chunk_lower = chunk.lower()
            has_theme = int(any(word in chunk_lower for word in theme_keywords))
            texts.append(chunk)
            labels.append(has_theme)
            if len(texts) >= n:
                break
        if len(texts) >= n:
            break
    return texts, labels

texts, labels = extract_samples_with_theme(n=200)
dataset = Dataset.from_dict({"text": texts, "label": labels})
dataset = dataset.train_test_split(test_size=0.2, seed=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
training_args = TrainingArguments(
    output_dir="./results_thematic",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs_thematic",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()

preds = trainer.predict(tokenized_dataset["test"])
y_true = preds.label_ids
y_pred = preds.predictions.argmax(-1)
print(classification_report(y_true, y_pred, target_names=["No Theme", "Theme"]))