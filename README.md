# Document Q&A System (RAG)

A command-line tool that answers questions about a PDF document using Retrieval-Augmented Generation (RAG). Ask a question, and it finds the most relevant parts of the document and generates an accurate, grounded answer — instead of requiring you to read the whole document yourself.

## How it works

1. **Load** — Extracts raw text from a PDF (`loader.py`)
2. **Chunk** — Splits the text into smaller overlapping pieces, so context isn't lost at boundaries (`chunker.py`)
3. **Embed** — Converts each chunk into a vector (embedding) representing its meaning, using OpenAI's `text-embedding-3-small` model (`embedder.py`)
4. **Store** — Saves all chunks and their embeddings into a local vector database using ChromaDB (`store.py`)
5. **Retrieve** — Converts a user's question into an embedding, and finds the most semantically similar stored chunks (`retriever.py`)
6. **Generate** — Sends the retrieved chunks + question to OpenAI's `gpt-4o-mini` model, with explicit instructions to only answer using the provided context (`generator.py`)

## Example