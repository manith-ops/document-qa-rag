from loader import load_pdf_text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

if __name__ == "__main__":
    text = load_pdf_text("Mancherial.pdf")
    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")
    print("First chunk:")
    print(chunks[0])