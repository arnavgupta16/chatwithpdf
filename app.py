import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Set page configuration
st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")

# Initialize session state variables
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Title and description
st.title("📚 PDF Question-Answering Assistant")
st.markdown("""
Ask questions about the document and get AI-powered answers based on the content.
""")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Groq API Key:", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

# Initialize the necessary components
def initialize_qa_system():
    # Specify your PDF path
    pdf_path = "pdf-sample.pdf"  # Replace with your actual PDF path
    
    # Load and process the PDF
    with st.spinner('Loading and processing the PDF...'):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        # Initialize LLM
        llm = ChatGroq(model_name="llama-3.1-8b-instant")

        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain

# Main app layout
if not st.session_state.qa_chain:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar first!")
    else:
        st.session_state.qa_chain = initialize_qa_system()
        st.success("System initialized successfully! You can now ask questions.")

# Question input and response
if st.session_state.qa_chain:
    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner('Searching for answer...'):
            try:
                result = st.session_state.qa_chain({"query": question})
                
                # Display answer
                st.markdown("### Answer:")
                st.write(result["result"])
                
                # Display sources
                st.markdown("### Sources:")
                for doc in result["source_documents"]:
                    st.write(f"- Page {doc.metadata['page']}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")