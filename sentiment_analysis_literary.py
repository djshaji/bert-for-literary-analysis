"""
Sentiment Analysis on Literary Texts with BERT

This script fine-tunes a BERT model for sentiment analysis on public domain literary passages.
Keyword heuristics are used for weak labeling (positive/negative/neutral).

Requires: transformers, datasets, nltk, torch, scikit-learn
Run: python sentiment_analysis_literary.py
"""
import nltk
from nltk.corpus import gutenberg
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report

nltk.download('gutenberg')

positive_words = ["love", "joy", "happy", "delight", "pleasure", "smile", "laugh", "good", "beautiful"]
negative_words = ["hate", "death", "sad", "fear", "pain", "cry", "bad", "ugly", "anger"]

def get_sentiment(chunk):
    c = chunk.lower()
    pos = sum(w in c for w in positive_words)
    neg = sum(w in c for w in negative_words)
    if pos > neg:
        return 2  # positive
    elif neg > pos:
        return 0  # negative
    else:
        return 1  # neutral

def extract_sentiment_samples(n=200):
    texts, labels = [], []
    for fileid in gutenberg.fileids():
        text = gutenberg.raw(fileid)
        chunks = [text[i:i+1800] for i in range(0, len(text), 1800)]
        for chunk in chunks:
            sentiment = get_sentiment(chunk)
            texts.append(chunk)
            labels.append(sentiment)
            if len(texts) >= n:
                break
        if len(texts) >= n:
            break
    return texts, labels

texts, labels = extract_sentiment_samples(n=200)
dataset = Dataset.from_dict({"text": texts, "label": labels})
dataset = dataset.train_test_split(test_size=0.2, seed=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
training_args = TrainingArguments(
    output_dir="./results_sentiment",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs_sentiment",
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
print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]))