from openai import OpenAI          # the official OpenAI library, lets us call their API
from dotenv import load_dotenv     # reads our .env file so we can access the API key safely
from chunker import chunk_text     # reuse our chunking function from chunker.py
from loader import load_pdf_text   # reuse our PDF-loading function from loader.py

load_dotenv()  # actually loads the .env file's contents into the environment, so Python can read OPENAI_API_KEY

client = OpenAI()  
# creates a "client" object — this is our connection to OpenAI's API
# it automatically looks for OPENAI_API_KEY in the environment (which load_dotenv() just loaded) — we never type the key directly here

def get_embedding(text):
    # takes one piece of text, returns its embedding (a list of numbers representing meaning)
    
    response = client.embeddings.create(
        model="text-embedding-3-small",  
        # this is OpenAI's embedding model — "small" version is cheaper and fast, good enough for our project
        input=text  
        # the actual text we want converted into a vector
    )
    
    return response.data[0].embedding  
    # the API returns a structured response; this line digs into it and pulls out just the actual vector (list of numbers) we want

if __name__ == "__main__":
    text = load_pdf_text("Mancherial.pdf")   # get the full PDF text
    chunks = chunk_text(text)                 # split into chunks
    
    first_embedding = get_embedding(chunks[0])  
    # test: generate an embedding for just the FIRST chunk (not all 33 yet — we're testing this works before scaling up)
    
    print(f"Embedding length: {len(first_embedding)}")  
    # embeddings are typically 1536 numbers long for this model — this confirms it worked
    print(f"First few numbers: {first_embedding[:5]}")  
    # just peek at the first 5 numbers, so we can see it's actually a list of numbers, not gibberish