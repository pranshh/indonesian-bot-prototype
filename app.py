from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import yaml
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from PyPDF2 import PdfReader

# --- Basic Setup ---
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'docs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Groq API and Embedding Model Configuration ---

# Get the absolute path to the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEYS_YAML_PATH = os.path.join(BASE_DIR, "keys.yaml")

try:
    # Load API key from the YAML file using the absolute path
    with open(KEYS_YAML_PATH, "r") as file:
        keys = yaml.safe_load(file)
    
    # Safely get the API key with better error handling
    groq_api_key = keys.get("groq", {}).get("api_key")
    if not groq_api_key:
        raise ValueError("API key for 'groq' not found in keys.yaml")

    groq_client = Groq(api_key=groq_api_key)
    GROQ_MODEL = "llama3-8b-8192"
    print("Groq client initialized successfully.")

except FileNotFoundError:
    print(f"ERROR: keys.yaml file not found at {KEYS_YAML_PATH}")
    groq_client = None
except (ValueError, AttributeError, KeyError) as e:
    print(f"ERROR: Could not read API key from keys.yaml. Please check its structure. Details: {e}")
    groq_client = None
except Exception as e:
    print(f"Failed to initialize Groq client: {e}")
    groq_client = None


# Embedding model for document retrieval
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_db = None

# --- Core Functions ---

def pdf_to_text(pdf_path):
    """Extracts text from a single PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        print(f"Successfully extracted text from {os.path.basename(pdf_path)}.")
        return text
    except Exception as e:
        print(f"Error reading PDF {os.path.basename(pdf_path)}: {e}")
        return None

def retrieve_context(query, db, k=5):
    """Retrieves the most relevant text chunks from the vector DB."""
    if not db:
        return ""
    docs = db.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    print(f"Retrieved context for query: '{query}'")
    return context

# --- Flask Routes ---

@app.route("/", methods=["GET"])
def index():
    """Renders the main HTML page."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Handles multiple PDF file uploads and builds/updates the vector DB."""
    global vector_db
    uploaded_files = request.files.getlist("files")
    if not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == ''):
        return jsonify({"error": "No files selected"}), 400

    all_docs = []
    processed_files = []
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for file in uploaded_files:
        if file and file.filename.endswith('.pdf'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            text = pdf_to_text(filepath)
            if text:
                docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
                all_docs.extend(docs)
                processed_files.append(file.filename)

    if not all_docs:
        return jsonify({"error": "Could not extract any text from the provided PDF(s)."}), 500

    if vector_db:
        vector_db.add_documents(all_docs)
        message = f"Added new documents to knowledge base: {', '.join(processed_files)}"
        print("Added new documents to the existing vector database.")
    else:
        vector_db = FAISS.from_documents(all_docs, embeddings)
        message = f"PDF(s) '{', '.join(processed_files)}' uploaded and indexed!"
        print("Created a new vector database.")
    return jsonify({"message": message})

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat requests using the Groq API."""
    if not vector_db:
        return jsonify({"error": "Please upload one or more PDF documents first."}), 400
    if not groq_client:
        return jsonify({"error": "Groq client is not available. Check server logs for API key issues."}), 500
        
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Message cannot be empty."}), 400

    context = retrieve_context(user_input, vector_db)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Indonesian assistant. Your primary role is to answer questions based *strictly* on the context provided by the user. "
                "Follow these rules precisely:\n"
                "1. Your entire response must be in Bahasa Indonesia.\n"
                "2. Base your answer solely on the information within the 'Context' section. Do not use any external knowledge.\n"
                "3. If the answer is not found in the context, you must state 'Maaf, informasi tersebut tidak ditemukan dalam dokumen yang diberikan.' and nothing else.\n"
                "4. Keep your answers concise and directly address the user's question."
                "5. Do not mention 'Menurut dokumen Travel Policy' for every response. Just write your response normally"
            )
        },
        {
            "role": "user",
            "content": f"Context:\n---\n{context}\n---\n\nQuestion: {user_input}"
        }
    ]
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            temperature=0.5,
            max_tokens=1024,
        )
        response_content = chat_completion.choices[0].message.content
        return jsonify({"response": response_content.strip()})
        
    except Exception as e:
        print(f"Groq API Error: {e}")
        # This is where the "limit reached" error from Groq will likely appear
        return jsonify({"error": f"An error occurred with the Groq API: {e}"}), 500

# --- Main Execution ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)
