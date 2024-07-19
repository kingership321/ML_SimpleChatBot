import PyPDF2
import numpy as np
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

# Download necessary NLTK data
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def adaptive_chunking(text, max_chunk_size=300):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_embeddings(chunks, model):
    return model.encode(chunks)

def two_stage_retrieval(question, chunks, chunk_embeddings, model, k=3):
    question_embedding = model.encode([question])[0]
    similarities = cosine_similarity(question_embedding.reshape(1, -1), chunk_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    top_chunks = [chunks[i] for i in top_k_indices]
    sentences = [sent for chunk in top_chunks for sent in nltk.sent_tokenize(chunk)]
    sentence_embeddings = model.encode(sentences)
    
    sentence_similarities = cosine_similarity(question_embedding.reshape(1, -1), sentence_embeddings)[0]
    best_sentence_idx = np.argmax(sentence_similarities)
    
    return sentences[best_sentence_idx]

def classify_question(question):
    doc = nlp(question)
    for token in doc:
        if token.tag_ == "WP" or token.tag_ == "WRB":
            return "wh-question"
    return "other"

# Main script to interact with the document
def main():
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    pdf_path = r"C:\Users\DELL\Desktop\InternshipReport.pdf"  # Change this to your PDF file path
    text = extract_text_from_pdf(pdf_path)
    chunks = adaptive_chunking(text)
    chunk_embeddings = create_embeddings(chunks, model)
    
    while True:
        question = input("Ask your question (type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        question_type = classify_question(question)
        if question_type == "wh-question":
            answer = two_stage_retrieval(question, chunks, chunk_embeddings, model)
            print("Answer:", answer)
        else:
            print("Please ask a wh-question (e.g., who, what, when, where, why, how).")

if __name__ == "__main__":
    main()