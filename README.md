# Semantic Chunking & Embedding Tool

This Python tool processes text documents and splits them into semantically meaningful chunks using three strategies:
- Fixed-size with overlap
- Sentence-based
- Paragraph-based

Each chunk is embedded using a modern sentence embedding model (`all-MiniLM-L6-v2`), stored in a local FAISS vector index, and used for semantic search based on cosine similarity.

## Features

- **Supports** `.txt`, `.pdf`, and `.docx` files  
- **Implements** multiple chunking strategies  
- **Performs** semantic search across all chunks  
- **Automatically selects** the most relevant strategy  
- **Stores** embeddings using FAISS for fast retrieval  
- **Saves** matching text chunks for reuse  

## Output Files

- `semantic_index.faiss` – vector database of embeddings  
- `semantic_chunks.pkl` – corresponding text segments  

## Included Sample Files

This repository contains **3 sample text documents** for testing the model: `Ai.txt`, `what_is_machine_learning.docx` and `climate_change.pdf`
