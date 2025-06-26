import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


class CustomAPIEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embddings: list[list[float]] = []
        for text in texts:
            embedding = (
                requests.post(
                    "http://10.31.3.55:11434/api/embed",
                    json={"model": "nomic-embed-text:v1.5", "input": text},
                )
                .json()
                .get("embeddings", [])
            )
            embddings.append(embedding[0])
        return embddings

    def embed_query(self, text: str) -> list[float]:
        embedding = (
            requests.post(
                "http://10.31.3.55:11434/api/embed",
                json={"model": "nomic-embed-text:v1.5", "input": text},
            )
            .json()
            .get("embeddings", [])
        )
        return embedding[0]


def get_document_contents(path: str) -> str:
    with open(path, "r") as file:
        return file.read()


document_contents = get_document_contents("vinland-saga.txt")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256, chunk_overlap=64, add_start_index=True
)
chunks = text_splitter.split_text(document_contents)


def genrate_documents(chunks: list[str]) -> list[Document]:
    documents: list[Document] = []
    for chunk in chunks:
        documents.append(
            Document(page_content=chunk, metadata={"source": "vinland-saga.txt"})
        )
    return documents


documents = genrate_documents(chunks)

vector_store = InMemoryVectorStore(CustomAPIEmbeddings())
vector_store.add_documents(documents=documents)


results = vector_store.similarity_search(query="Who is the writer of Vinland Saga?")

print(results[0].page_content)
