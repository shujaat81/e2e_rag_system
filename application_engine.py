# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        # Retrieve relevant documents using embeddings
        retrieved_data = self.retriever.query(query_texts=question, n_results=5)

        # Extract document contents from the retrieved metadata
        doc_texts = "\\n".join(
            [meta["text"] for meta in retrieved_data["metadatas"][0]]
        )

        # Pass the concatenated document text to the language model for answering the question
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})

        return answer
