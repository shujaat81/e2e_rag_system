from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from chromadb.config import Settings
from rag_classes import RAGApplication
from langchain_core.output_parsers import StrOutputParser
from utils import get_model, get_prompt_template

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize ChromaDB client and collection
chroma_client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings(allow_reset=True, anonymized_telemetry=False),
)
document_collection = chroma_client.get_collection(name="document_semantic")

# Initialize the LLM model and prompt
llm = get_model()
prompt = get_prompt_template()

# Create the RAG chain
rag_chain = prompt | llm | StrOutputParser()

# Initialize RAG Application
rag_application = RAGApplication(document_collection, rag_chain)


# Define an endpoint for the RAG application
@app.route("/ask", methods=["POST"])
def ask_question():
    # Get the question from the incoming request
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Run the RAG application to get the answer
    try:
        answer = rag_application.run(question)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Define a health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
