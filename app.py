from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers

# Load PDF
loader = PyPDFLoader(r"data/java_notes.pdf")
documents = loader.load()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# Create embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
vectorstore = FAISS.from_documents(
    docs,
    embedding
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Load FREE local LLM
llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGUF",
    model_type="llama",
    config={
        'max_new_tokens': 256,
        'temperature': 0.01
    }
)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Ask user question
query = input("Ask a Java Question: ")

# Generate answer
response = qa.run(query)

print("\nAnswer:\n")
print(response)