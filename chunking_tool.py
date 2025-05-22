import os
import fitz  
import docx
import nltk
import numpy as np
import faiss
import pickle
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')


# Load document
def load_document(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' not found. Please check the name and extension.")

    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif path.endswith(".pdf"):
        text = ""
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        return text.replace("\n", " ")
    elif path.endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format. Please use .txt, .pdf, or .docx.")


# Chunking strategies
def chunk_fixed(text, size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks

def chunk_sentences(text):
    return sent_tokenize(text)

def chunk_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# Embedding & similarity
def create_embeddings(texts, model):
    embeddings = model.encode(texts)
    return embeddings

def search_similar_cosine(embeddings, texts, query, model, top_k=1):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(texts[i], similarities[i]) for i in top_indices]


# Combine all strategies and pick best result + save FAISS index and texts
def get_best_result(text, query, model):
    # Generate chunks
    chunks_fixed = chunk_fixed(text)
    chunks_sent = chunk_sentences(text)
    chunks_para = chunk_paragraphs(text)

    # Generate embeddings
    emb_fixed = create_embeddings(chunks_fixed, model)
    emb_sent = create_embeddings(chunks_sent, model)
    emb_para = create_embeddings(chunks_para, model)

    # Search in each
    result_fixed = search_similar_cosine(emb_fixed, chunks_fixed, query, model)[0]
    result_sent = search_similar_cosine(emb_sent, chunks_sent, query, model)[0]
    result_para = search_similar_cosine(emb_para, chunks_para, query, model)[0]

    all_results = {
        "fixed": result_fixed,
        "sentences": result_sent,
        "paragraphs": result_para
    }

    # Pick highest score
    best_strategy, best_result = max(all_results.items(), key=lambda x: x[1][1])
    best_text, score = best_result

    # Select corresponding embeddings and chunks
    if best_strategy == "fixed":
        selected_embeddings = emb_fixed
        selected_chunks = chunks_fixed
    elif best_strategy == "sentences":
        selected_embeddings = emb_sent
        selected_chunks = chunks_sent
    else:
        selected_embeddings = emb_para
        selected_chunks = chunks_para

    # Create and save FAISS index
    dimension = len(selected_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(selected_embeddings).astype("float32"))
    faiss.write_index(index, "semantic_index.faiss")

    # Save text chunks to pickle
    with open("semantic_chunks.pkl", "wb") as f:
        pickle.dump(selected_chunks, f)

    return best_strategy, best_result, all_results


# Main interface
def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    text = None
    path = None

    while True:
        if not text:
            path = input("Enter the file name: ").strip()
            try:
                text = load_document(path)
            except Exception as e:
                print("Error:", e)
                continue

        query = input("\nEnter a query: (Enter 'exit' to quit | Enter 'change' to load another document): ").strip()

        if query.lower() == "exit":
            break
        elif query.lower() == "change":
            text = None
            continue

        best_strategy, (best_text, score), all_results = get_best_result(text, query, model)

        print(f"\nBest matching result (strategy: {best_strategy}, score: {score:.2f}):\n")
        if score < 0.6:
            print("Low semantic similarity – the result may not be accurate.\n")
        print(best_text.strip())


if __name__ == "__main__":
    main()
