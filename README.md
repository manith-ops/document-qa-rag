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

Question: What is the population of Mancherial?
Answer: The population of Mancherial is 228,450.

## Tech stack

- Python
- OpenAI API (embeddings + chat completion)
- ChromaDB (vector database)
- pypdf (PDF text extraction)
- python-dotenv (secure API key management)

## Setup

1. Clone this repo
2. Create a virtual environment: `python3 -m venv venv` then `source venv/bin/activate`
3. Install dependencies: `pip install openai pypdf chromadb python-dotenv`
4. Create a `.env` file with your OpenAI API key: `OPENAI_API_KEY=your-key-here`
5. Run `python3 store.py` once, to build the vector database from the PDF
6. Run `python3 generator.py` to ask a question (edit the `question` variable inside the file)

## What I learned

Building this project gave me hands-on understanding of the core RAG architecture used in real production AI systems — chunking strategy, embeddings, vector search, and prompt grounding to prevent hallucination. Coming from a Data Engineering background, this project let me apply my existing experience with data pipelines (extract → transform → load) to an AI-specific context.

## Possible future improvements

- Interactive loop for asking multiple questions without editing code
- Support for multiple documents
- Web interface instead of command-line