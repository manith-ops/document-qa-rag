from pypdf import PdfReader

def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    full_text = ""
    
    for page in reader.pages:
        full_text += page.extract_text()
    
    return full_text

if __name__ == "__main__":
    text = load_pdf_text("Mancherial.pdf")
    print(text)