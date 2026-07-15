from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(page_content="The Eiffel Tower is located in Paris, France."),
    Document(page_content="Python was created by Guido van Rossum in 1991."),
    Document(page_content="LangChain is a framework for building LLM applications."),
]

embeddings = FakeEmbeddings(size=384)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def answer_question(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    response = llm.invoke(f"Answer based on this context only:\n{context}\n\nQuestion: {question}")
    return response.content