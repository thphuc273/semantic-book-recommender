import pandas as pd
import numpy as np
from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# =======================
# Load and preprocess books
# =======================
books = pd.read_csv("data/books_with_emotions.csv")

books["large_thumbnail"] = np.where(
    books["thumbnail"].notna(),
    books["thumbnail"] + "&fife=w800",
    "cover-not-found.jpg"
)

# =======================
# Prepare Chroma vector DB
# =======================
raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

# =======================
# Semantic retrieval logic
# =======================
def retrieve_semantic_recommendations(query: str,
                                      category: str = "All",
                                      tone: str = "All",
                                      initial_top_k: int = 50,
                                      final_top_k: int = 16) -> pd.DataFrame:
    """Truy xuất danh sách gợi ý dựa trên ngữ nghĩa, danh mục và cảm xúc."""

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]

    # Lọc sách theo ISBN
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # Lọc theo category 
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # Sắp xếp theo tone cảm xúc
    tone_sort_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }
    if tone in tone_sort_map:
        book_recs = book_recs.sort_values(by=tone_sort_map[tone], ascending=False)

    return book_recs.head(final_top_k)

# =======================
# Recommendation formatting
# =======================
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        desc = row["description"].split()
        truncated_description = " ".join(desc[:30]) + "..."

        authors = row["authors"].split(";")
        if len(authors) == 1:
            authors_str = authors[0]
        elif len(authors) == 2:
            authors_str = f"{authors[0]} and {authors[1]}"
        else:
            authors_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# =======================
# Build Gradio dashboard
# =======================
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness"
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="All"
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## 🧠 Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()
