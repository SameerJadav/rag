import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_document_contents(path: str) -> str:
    with open(path, "r") as file:
        return file.read()


document_contents = get_document_contents("vinland-saga.txt")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256, chunk_overlap=64, add_start_index=True
)
chunks = text_splitter.split_text(document_contents)

documents = [
    Document(page_content=chunk, metadata={"source": "vinland-saga.txt"})
    for chunk in chunks
]

vector_store = Chroma(
    embedding_function=OllamaEmbeddings(model="nomic-embed-text:v1.5")
)
vector_store.add_documents(documents=documents)

query = "Who is the writer of Vinland Saga?"
# query = "Who is the writer and illustrator of Vinland Saga?"
# query = "Who is the main character in Vinland Saga?"
# query = "Which manga has a character inspired by King Canute?"
# query = "write a hello world program in python"
# query = "hii"

docs_and_scores = vector_store.similarity_search_with_relevance_scores(
    query=query, k=5, score_threshold=0.50
)

if len(docs_and_scores) == 0:
    print("low relevance score.")
else:
    context = "\n".join([doc_and_score[0].page_content for doc_and_score in docs_and_scores])


    prompt = f"""You are a helpful AI assistant that answers questions based on the provided PDF document. Use only the context provided to answer the question. If you don't know the answer or can't find it in the context, say so.

Context:
{context}

Question:
{query}

Answer:
Let me help you with that based on the PDF content."""
    
    def query_llm(prompt: str) -> str:
        res = requests.post("http://10.11.5.136:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        return res.json().get("response", "")
    
    print(query_llm(prompt))