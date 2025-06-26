from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
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


def genrate_documents(chunks: list[str]) -> list[Document]:
    documents: list[Document] = []
    for chunk in chunks:
        documents.append(
            Document(page_content=chunk, metadata={"source": "vinland-saga.txt"})
        )
    return documents


documents = genrate_documents(chunks)

vector_store = InMemoryVectorStore(OllamaEmbeddings(model="nomic-embed-text:v1.5"))
vector_store.add_documents(documents=documents)

query = "Who is the writer of Vinland Saga?"
# query = "Who is the writer and illustrator of Vinland Saga?"
# query = "Who is the main character in Vinland Saga?"
# query = "Which manga has a character inspired by King Canute?"
results = vector_store.similarity_search(query=query)

for result in results:
    print(result.page_content)
    print()
