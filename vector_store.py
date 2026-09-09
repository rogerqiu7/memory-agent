from pathlib import Path
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


DATA_FILE = Path("data/realistic_restaurant_reviews.csv")
DB_LOCATION = Path("chroma_langchain_db")
IS_NEW_STORE = not DB_LOCATION.exists()
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=str(DB_LOCATION),
    embedding_function=embeddings,
)

if IS_NEW_STORE:
    reviews = pd.read_csv(DATA_FILE)
    documents = [
        Document(
            page_content=f"{row.Title} {row.Review}",
            metadata={"rating": row.Rating, "date": row.Date},
        )
        for row in reviews.itertuples()
    ]
    vector_store.add_documents(
        documents=documents,
        ids=[str(index) for index in range(len(documents))],
    )

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
