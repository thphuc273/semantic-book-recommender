import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# Initialize the emotion classifier
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device="mps"
)

books = pd.read_csv("data/book_with_categories.csv")

# Define emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

def calculate_max_emotion_score(predictions):
    """Calculate maximum score for each emotion from predictions."""
    per_emotion_scores = {label: [] for label in emotion_labels}
    for pred in predictions:
        sorted_pred = sorted(pred, key=lambda x: x['label'])
        for idx, label in enumerate(emotion_labels):
            per_emotion_scores[label].append(sorted_pred[idx]['score'])
    return {label: np.max(scores) for label, scores in per_emotion_scores.items()}

# Process sentiment analysis for all books
emotion_scores = {label: [] for label in emotion_labels}
isbn = []

for i in tqdm(range(len(books)), desc="Processing books"):
    isbn.append(books.iloc[i]['isbn13'])
    sentences = books.iloc[i]['description'].split('.')
    predictions = classifier(sentences)
    max_scores = calculate_max_emotion_score(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])


emotions_df = pd.DataFrame(emotion_scores)
emotions_df["isbn13"] = isbn

# Merge with original books DataFrame
books = pd.merge(books, emotions_df, on="isbn13")

books.to_csv('data/books_with_emotions.csv', index=False)

print("Sentiment analysis completed and saved to 'data/books_with_emotions.csv'")