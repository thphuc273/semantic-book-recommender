import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# Load data
books = pd.read_csv('data/book_cleaned.csv')

# Define category mapping for simplification
categories_mapping = {
    'Fiction': 'Fiction',
    'Juvenile Fiction': 'Fiction',
    'Biography & Autobiography': 'Nonfiction',
    'History': 'Nonfiction',
    'Literary Criticism': 'Nonfiction',
    'Philosophy': 'Nonfiction',
    'Religion': 'Nonfiction',
    'Comics & Graphic Novels': 'Fiction',
    'Drama': 'Fiction',
    'Juvenile Nonfiction': 'Nonfiction',
    'Science': 'Nonfiction',
    'Poetry': 'Fiction'
}

# Apply category mapping
books['simple_categories'] = books['categories'].map(categories_mapping)

# Initialize zero-shot classifier
fiction_categories = ['Fiction', 'Nonfiction']
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device="mps"
)

def generate_prediction(sequence: str, categories: list, classifier: pipeline) -> str:
    """
    Generate predicted category for a given sequence using a zero-shot classifier.

    Args:
        sequence (str): Input text to classify.
        categories (list): List of possible categories.
        classifier (pipeline): Hugging Face zero-shot classification pipeline.

    Returns:
        str: Predicted category label.

    Raises:
        ValueError: If sequence is empty or invalid.
        RuntimeError: If classifier output is invalid.
    """
    if not sequence or not isinstance(sequence, str):
        raise ValueError("Sequence must be a non-empty string")
    if not categories or not isinstance(categories, list):
        raise ValueError("Categories must be a non-empty list")

    try:
        sequence = sequence[:512] if len(sequence) > 512 else sequence
        result = classifier(sequence, candidate_labels=categories, multi_label=False)
        if not isinstance(result, dict) or 'labels' not in result or 'scores' not in result:
            raise RuntimeError(f"Unexpected classifier output: {result}")
        max_idx = np.argmax(result['scores'])
        return result['labels'][max_idx]
    except Exception as e:
        raise RuntimeError(f"Error in prediction: {str(e)}")

# Evaluate classifier on a sample of known categories
actual_cats = []
pred_cats = []

# Process Fiction samples
fiction_descriptions = books.loc[books['simple_categories'] == "Fiction", "description"].reset_index(drop=True)
for i in tqdm(range(min(200, len(fiction_descriptions))), desc="Processing Fiction"):
    try:
        actual_cats.append("Fiction")
        pred_cats.append(generate_prediction(fiction_descriptions[i], fiction_categories, classifier))
    except Exception as e:
        print(f"Error processing Fiction sample {i}: {str(e)}")
        pred_cats.append("Unknown")  # Fallback category

# Process Nonfiction samples
nonfiction_descriptions = books.loc[books['simple_categories'] == "Nonfiction", "description"].reset_index(drop=True)
for i in tqdm(range(min(200, len(nonfiction_descriptions))), desc="Processing Nonfiction"):
    try:
        actual_cats.append("Nonfiction")
        pred_cats.append(generate_prediction(nonfiction_descriptions[i], fiction_categories, classifier))
    except Exception as e:
        print(f"Error processing Nonfiction sample {i}: {str(e)}")
        pred_cats.append("Unknown")  # Fallback category

# Create predictions DataFrame and calculate accuracy
preds_df = pd.DataFrame({"actual_cats": actual_cats, "pred_cats": pred_cats})
preds_df["correct_pred"] = (preds_df["actual_cats"] == preds_df["pred_cats"]).astype(int)
accuracy = preds_df["correct_pred"].mean()
print(f"Classification accuracy: {accuracy:.4f}")

# Predict categories for missing values
missing_cat = books.loc[books["simple_categories"].isna(), ["isbn13", "description"]].reset_index(drop=True)
isbn = []
preds = []

for i in tqdm(range(len(missing_cat)), desc="Predicting missing categories"):
    try:
        isbn.append(missing_cat['isbn13'][i])
        preds.append(generate_prediction(missing_cat['description'][i], fiction_categories, classifier))
    except Exception as e:
        print(f"Error predicting for ISBN {missing_cat['isbn13'][i]}: {str(e)}")
        preds.append("Unknown")  # Fallback category
        isbn.append(missing_cat['isbn13'][i])

# Create DataFrame for predicted categories
missing_preds_df = pd.DataFrame({"isbn13": isbn, "predicted_categories": preds})

# Merge predictions and fill missing categories
books = pd.merge(books, missing_preds_df, on="isbn13", how="left")
books["simple_categories"] = books["simple_categories"].fillna(books["predicted_categories"])
books = books.drop(columns=["predicted_categories"])

# Save updated DataFrame
books.to_csv('data/book_with_categories.csv', index=False)

print("Category classification completed and saved to 'data/book_with_categories.csv'")
print("Category distribution:")
print(books['simple_categories'].value_counts())