import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

chroma_client = chromadb.PersistentClient(path="./chroma_db")  
# connect to the SAME database folder we created in store.py — not a new one, we're reading from what we already saved

collection = chroma_client.get_or_create_collection(name="mancherial_docs")  
# connect to the SAME collection name we used before, so we're reading the actual data we stored

def retrieve_relevant_chunks(question, n_results=3):
    # question = what the user is asking | n_results = how many top matching chunks to return
    
    question_embedding = get_embedding(question)  
    # convert the QUESTION into an embedding too — same process, same function, just different input
    # this is the key idea: question and document chunks both become numbers, so we can compare them directly
    
    results = collection.query(
        query_embeddings=[question_embedding],  
        # search using this embedding
        n_results=n_results  
        # only give us the top 3 closest matches, not all 33
    )
    
    return results['documents'][0]  
    # results comes back in a nested structure; this digs in and returns just the actual text of the matching chunks

if __name__ == "__main__":
    question = "What is the population of Mancherial?"  
    # test question — swap this for anything relevant to your actual PDF's content later
    
    relevant_chunks = retrieve_relevant_chunks(question)
    
    print(f"Question: {question}\n")
    print("Top relevant chunks found:")
    for i, chunk in enumerate(relevant_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)