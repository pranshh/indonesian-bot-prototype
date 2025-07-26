from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import yaml
import google.generativeai as genai
from google.cloud import texttospeech
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
import base64

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'docs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEYS_YAML_PATH = os.path.join(BASE_DIR, "keys.yaml")

try:
    with open(KEYS_YAML_PATH, "r") as file:
        keys = yaml.safe_load(file)

    gemini_api_key = keys.get("gemini", {}).get("api_key")
    if not gemini_api_key:
        raise ValueError("API key for 'gemini' not found in keys.yaml")

    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash") # Or "gemini-pro"
    print("Gemini client initialized successfully.")

except FileNotFoundError:
    print(f"ERROR: keys.yaml file not found at {KEYS_YAML_PATH}")
    groq_client = None
except (ValueError, AttributeError, KeyError) as e:
    print(f"ERROR: Could not read API key from keys.yaml. Please check its structure. Details: {e}")
    groq_client = None
except Exception as e:
    print(f"Failed to initialize Gemini client: {e}")
    groq_client = None

try:
    tts_client = texttospeech.TextToSpeechClient()
    print("✅ Google Cloud TTS client initialized successfully.")
except Exception as e:
    print(f"❌ Could not initialize Google Cloud TTS client. Make sure GOOGLE_APPLICATION_CREDENTIALS is set. Error: {e}")
    tts_client = None

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_db = None

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
    """Handles chat requests using the Gemini SDK."""
    if not vector_db:
        return jsonify({"error": "Please upload one or more PDF documents first."}), 400
    if not gemini_model:
        return jsonify({"error": "Gemini client is not available. Check server logs for API key issues."}), 500
    
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Message cannot be empty."}), 400

    context = retrieve_context(user_input, vector_db)
    
    prompt = (
        "You are an expert Indonesian assistant. Your primary role is to answer questions based *strictly* on the context provided by the user. "
        "Follow these rules precisely:\n"
        "1. Your entire response must be in Bahasa Indonesia.\n"
        "2. Base your answer solely on the information within the 'Context' section. Do not use any external knowledge.\n"
        "3. If the answer is not found in the context, you must state 'Maaf, informasi tersebut tidak ditemukan dalam dokumen yang diberikan.' and nothing else.\n"
        "4. Keep your answers concise and directly address the user's question.\n"
        "5. Do not mention 'Menurut dokumen Travel Policy' for every response. Just write your response normally.\n\n"
        f"Context:\n---\n{context}\n---\n\n"
        f"Question: {user_input}"
    )
    
    try:
        response = gemini_model.generate_content(prompt)
        text_response = response.text.strip()
        
        audio_base64 = None
        if tts_client:
            try:
                synthesis_input = texttospeech.SynthesisInput(text=text_response)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="id-ID", name="id-ID-Standard-A"
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                tts_response = tts_client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )

                audio_base64 = base64.b64encode(tts_response.audio_content).decode("utf-8")
            except Exception as e:
                print(f"TTS synthesis failed: {e}")
                pass
        else:
            print("TTS client is not initialized. Skipping audio generation.")
        return jsonify({"response": text_response, "audio": audio_base64})

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return jsonify({"error": f"An error occurred with the Gemini API: {e}"}), 500

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)