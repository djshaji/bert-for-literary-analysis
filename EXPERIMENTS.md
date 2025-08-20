# Example Experiments

## 1. Stylistic Classification

- **Task**: Classify samples as "Austen" or "Shakespeare".
- **Data**: NLTK Gutenberg corpus.
- **Method**: Fine-tune BERT for sequence classification.
- **Metric**: Accuracy, F1.

## 2. Sentiment Analysis

- **Task**: Classify excerpts as "positive", "neutral", or "negative".
- **Data**: Annotate samples or use [Project Gutenberg Sentiment Dataset](https://huggingface.co/datasets/gutenberg_time) (if available).
- **Method**: Fine-tune BERT for sentiment.

## 3. Thematic Motif Detection

- **Task**: Detect presence/absence of themes (e.g., "love", "revenge") using keyword matching for labeling, then fine-tune BERT.

## 4. Authorship Attribution

- **Task**: Multi-class classification among several authors.

---

## Notes

- Fine-tuning on small literary datasets may require data augmentation or pre-training BERT on a large literary corpus for better domain adaptation.
- For more advanced motif detection, consider using multi-label classification and/or unsupervised topic modeling as auxiliary steps.