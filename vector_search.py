import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

books = pd.read_csv('data/book_with_categories.csv')

def retrieve_semantic_recommendations(query: str, top_k: int = 10, db_books=None) -> pd.DataFrame:
    """
    Retrieve top-k book recommendations based on semantic similarity to the query.
    
    Args:
        query (str): The search query.
        top_k (int): Number of recommendations to return.
        db_books: FAISS vector store containing book embeddings.
    
    Returns:
        pd.DataFrame: DataFrame with top-k book recommendations.
    """
    if db_books is None:
        raise ValueError("db_books vector store is required")

    # Perform similarity search
    recs = db_books.similarity_search(query, k=50)
    
    # Extract ISBNs from search results
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    
    # Filter books DataFrame and return top-k
    return books[books["isbn13"].isin(books_list)].head(top_k)

if __name__ == "__main__":
    books = pd.read_csv('data/book_cleaned.csv')
    books['tagged_description'].to_csv('data/tagged_description.txt', 
                                    sep='\n',
                                    index=False, 
                                    header=False)
    raw_docs = TextLoader('data/tagged_description.txt', encoding='utf-8').load()
    text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(raw_docs)

    embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    db_books = Chroma.from_documents(
    documents,
    embedding=embedding)  

    query = "A book to teach children about nature"

    recommendations = retrieve_semantic_recommendations(query, top_k=10, db_books=db_books)
    print(recommendations)