# Indonesian RAG Chatbot Prototype

A web-based chatbot that answers questions in Bahasa Indonesia using Retrieval-Augmented Generation (RAG) from your uploaded PDF documents. Powered by Google Gemini for language generation and Google Cloud Text-to-Speech for audio responses.

## Features

- **PDF Upload:** Upload one or more PDF files as the knowledge base.
- **Contextual Answers:** The bot answers strictly based on the uploaded documents.
- **Bahasa Indonesia Only:** All responses are in Bahasa Indonesia.
- **Text-to-Speech:** Bot replies include an audio version (MP3) using Google Cloud TTS.
- **Replay Audio:** Click the play button to replay bot responses.
- **Speech-to-Text Input:** Use your microphone to ask questions (browser support required).
- **Modern UI:** Clean, responsive interface with chat history.

## Requirements

- Python 3.8+
- Google Gemini API access and API key
- Google Cloud account with Text-to-Speech enabled and credentials JSON
- [HuggingFace Transformers](https://huggingface.co/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/indonesian-bot-prototype.git
   cd indonesian-bot-prototype
   ```

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Set up API keys:**

   - Create a `keys.yaml` file in the project root:

     ```yaml
     gemini:
       api_key: "YOUR_GEMINI_API_KEY"
     ```

   - Set the Google Cloud credentials environment variable:

     ```sh
     set GOOGLE_APPLICATION_CREDENTIALS=path\to\your\google-credentials.json
     ```

4. **Run the app:**

   ```sh
   python app.py
   ```

   The app will be available at [http://localhost:5000](http://localhost:5000).

## Usage

1. Open the web interface.
2. Upload one or more PDF files.
3. Ask questions in Bahasa Indonesia about the content of your PDFs.
4. Listen to the bot's answers using the audio play button.
5. Use the microphone button to ask questions by voice (if supported by your browser).

## File Structure

```text
indonesian-bot-prototype/
├── app.py                # Main Flask backend
├── keys.yaml             # Your Gemini API key (not included)
├── requirements.txt      # Python dependencies
├── docs/                 # Uploaded PDF files
├── templates/
│   └── index.html        # Main web UI
├── static/
│   └── style.css         # CSS styles
└── README.md             # This file
```

## Environment Variables

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your Google Cloud service account JSON file.

## Notes

- The bot will only answer based on the uploaded documents. If the answer is not found, it will reply:  
  *"Maaf, informasi tersebut tidak ditemukan dalam dokumen yang diberikan."*
- All responses are in Bahasa Indonesia.
- Make sure your Google Cloud project has Text-to-Speech enabled and billing set up.

## Troubleshooting

- **Gemini client not initialized:** Check your `keys.yaml` file and API key.
- **Google Cloud TTS not working:** Ensure `GOOGLE_APPLICATION_CREDENTIALS` is set and valid.
- **PDF not processed:** Make sure your PDFs are not encrypted or corrupted.
