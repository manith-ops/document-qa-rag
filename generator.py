from openai import OpenAI
from dotenv import load_dotenv
from retriever import retrieve_relevant_chunks  
# reuse the retrieval function we just proved works

load_dotenv()
client = OpenAI()

def generate_answer(question):
    relevant_chunks = retrieve_relevant_chunks(question)  
    # step 1: get the top 3 relevant chunks, exactly like before
    
    context = "\n\n".join(relevant_chunks)  
    # join the 3 chunks into one block of text, separated by blank lines, so the AI sees them as organized context

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:"""  
    # this is our actual prompt — we're explicitly instructing the model to stick to the provided context
    # this instruction matters a lot: without it, the model might use its own general knowledge instead of YOUR document

    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        # a solid, affordable chat model — good balance of quality and cost for this project
        messages=[
            {"role": "user", "content": prompt}  
            # we're sending our full prompt (context + question) as a single user message
        ]
    )
    
    return response.choices[0].message.content  
    # dig into the response structure and pull out just the actual answer text

if __name__ == "__main__":
    question = "Is Mancherial a city in Adilabad district?"
    answer = generate_answer(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")