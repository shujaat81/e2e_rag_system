from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate


def get_prompt_template():
    # Define the prompt template for the LLM
    return PromptTemplate(
        template="""You are an assistant for question-answering tasks.
        Use the following documents to answer the question.
        If you don't find answer in the document provided, just tell from your knowledge.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents"],
    )


def get_model():
    return ChatOllama(
        model="llama3.1",
        temperature=0,
    )
