import chromadb                    # our vector database
from openai import OpenAI
from dotenv import load_dotenv
from chunker import chunk_text
from loader import load_pdf_text

load_dotenv()
client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

chroma_client = chromadb.PersistentClient(path="./chroma_db")  
# "PersistentClient" means it saves to disk (a folder called chroma_db), not just memory — so data survives after the script ends
# "path" tells it exactly where to save that folder

collection = chroma_client.get_or_create_collection(name="mancherial_docs")  
# a "collection" in chromadb is like a table in a normal database — a named group of related data
# get_or_create means: use it if it already exists, otherwise make a new one — safe to re-run this script

if __name__ == "__main__":
    text = load_pdf_text("Mancherial.pdf")
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks):  
        # enumerate() gives us both the chunk's position number (i) AND the chunk itself, at the same time — instead of just looping through chunks alone
        
        embedding = get_embedding(chunk)  
        # convert this one chunk into its embedding — this happens 33 times total, once per chunk
        
        collection.add(
            ids=[str(i)],           # every entry needs a unique ID — we're just using the chunk's position number as text
            embeddings=[embedding], # the actual vector we just generated
            documents=[chunk]       # the original text, stored alongside its embedding, so we can retrieve the readable text later, not just numbers
        )
        
        print(f"Stored chunk {i+1}/{len(chunks)}")  
        # progress indicator, so we can see it's actually working as it goes through all 33

    print("All chunks embedded and stored successfully.")