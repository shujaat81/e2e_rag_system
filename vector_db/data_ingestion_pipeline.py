import os
from tqdm import tqdm
import chromadb
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb.config import Settings

NUMBER_OF_CHUNKS = 50

chroma_host = os.environ.get("CHROMA_HOST", "chroma")

chroma_client = chromadb.HttpClient(
    host=chroma_host,
    port=8000,
    settings=Settings(allow_reset=True, anonymized_telemetry=False),
)

# load documents
documents = SimpleDirectoryReader(input_files=["./document.txt"]).load_data()

# loads BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5", cache_folder="models/"
)
print("loaded embedded model")

splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=70, embed_model=embed_model
)

nodes = splitter.get_nodes_from_documents(documents)

document_collection = chroma_client.get_or_create_collection(name="document_semantic")
print("collection created")
count = 0

with tqdm(total=NUMBER_OF_CHUNKS, desc="Inserting chunks into ChromaDB") as pbar:
    # Process each node (chunk) and insert it into ChromaDB
    for node in nodes:
        text_chunk = node.text  # Extract chunked text
        embedding = embed_model.get_text_embedding(
            text_chunk
        )  # Generate embeddings for the chunk

        # Insert into ChromaDB (assuming each chunk has a unique ID and corresponding embedding)
        document_collection.add(
            ids=[str(node.id_)],  # Use node ID or some other unique identifier
            embeddings=[embedding],  # Insert the embedding
            metadatas=[
                {"text": text_chunk}
            ],  # Optionally store metadata like chunk text
        )
        count += 1
        pbar.update(1)
        if count == NUMBER_OF_CHUNKS:
            break

    print("All Chunks inserted into ChromaDB successfully!")
