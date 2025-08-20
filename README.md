# BERT-Based Literary Analysis Demos

This repository provides **ready-to-run Python scripts** for validating the use of fine-tuned BERT in literary analysis, using public domain texts from NLTK's Gutenberg corpus.

## Requirements

- Python 3.8+
- pip install torch transformers datasets scikit-learn nltk

## Scripts

### 1. Thematic Motif Detection

Detects the presence of a theme (e.g. "love") in literary passages.

```bash
python thematic_detection.py
```

### 2. Sentiment Analysis

Classifies literary passages as Negative, Neutral, or Positive.

```bash
python sentiment_analysis_literary.py
```

## Data

Uses public domain books from the `nltk.corpus.gutenberg` module (Austen, Shakespeare, etc).

## Notes

- Both scripts use keyword heuristics for weak labeling. For scholarly use, replace with hand-annotated or higher quality labels.
- For different themes or sentiment lexicons, edit the keyword lists in the scripts.
- Results will print a classification report after training and evaluation.