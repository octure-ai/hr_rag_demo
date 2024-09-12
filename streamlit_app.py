import os
import tempfile
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_aws import BedrockEmbeddings, AmazonKnowledgeBasesRetriever
from langchain.llms import OpenAI, Bedrock
from langchain.chains import RetrievalQA
from pathlib import Path
import jwt
import json
from datetime import datetime
import pandas as pd

load_dotenv()

API_URL = "http://localhost:8000"  # Update this to your FastAPI app's URL when deployed
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")  # Make sure this matches your FastAPI app's secret key


def get_user_vector_store_path(username):
    base_dir = Path("./persistent_vector_stores")
    user_dir = base_dir / username
    user_dir.mkdir(parents=True, exist_ok=True)
    return str(user_dir)


def save_document_metadata(username, filename, timestamp):
    metadata_file = Path(f"./persistent_vector_stores/{username}/document_metadata.json")
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = []

    metadata.append({
        "filename":filename,
        "timestamp":timestamp
    })

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)


def get_user_documents(username):
    metadata_file = Path(f"./persistent_vector_stores/{username}/document_metadata.json")
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            documents = json.load(f)
        # Convert ISO format to more readable format
        for doc in documents:
            doc['timestamp'] = datetime.fromisoformat(doc['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        return documents
    return []


def login_form():
    st.title("Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        response = requests.post(
            f"{API_URL}/token",
            data={"username":username, "password":password}
        )
        if response.status_code == 200:
            token = response.json()["access_token"]
            st.session_state.token = token
            st.session_state.logged_in = True

            # Decode the token to get the username
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
                st.session_state.username = payload.get("sub")
            except jwt.ExpiredSignatureError:
                st.error("Token has expired. Please log in again.")
                return
            except jwt.InvalidTokenError:
                st.error("Invalid token. Please log in again.")
                return

            st.success("Logged in successfully!")
            st.empty()
            st.rerun()
        else:
            st.error("Invalid username or password")


def register_form():
    st.title("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        response = requests.post(
            f"{API_URL}/register",
            json={"username":username, "password":password}
        )
        if response.status_code == 201:
            st.success("User registered successfully! Please log in.")
        else:
            st.error(f"Registration failed. {response.json().get('detail', 'Unknown error')}")


def get_api_key(key_name):
    return os.getenv(key_name) or st.secrets.get(key_name)


def process_document(file_path):
    loader = UnstructuredFileLoader(file_path)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(document)


def update_vector_store(texts, embeddings, persist_directory):
    if os.path.exists(persist_directory):
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        vector_store.add_documents(texts)
    else:
        vector_store = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    return vector_store


def setup_rag_pipeline(retriever, llm):
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'username' not in st.session_state:
        st.session_state.username = None

    if not st.session_state.logged_in:
        st.sidebar.title("Authentication")
        auth_option = st.sidebar.radio("Choose an option", ["Login", "Register"])
        if auth_option == "Login":
            login_form()
        else:
            register_form()
    else:
        st.title("RAG Pipeline with HR docs demo")

        # Custom CSS for styling
        st.markdown("""
        <style>
        .stDataFrame {
            font-size: 16px;
        }
        .stDataFrame td {
            padding: 10px;
        }
        .stDataFrame th {
            padding: 10px;
            text-align: left;
            background-color: #f0f2f6;
        }
        .document-icon {
            font-size: 20px;
            margin-right: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Display user's documents
        st.header("Your Processed Documents")
        user_documents = get_user_documents(st.session_state.username)
        if user_documents:
            # Create a DataFrame for the table
            df = pd.DataFrame(user_documents)
            df['Document'] = df['filename'].apply(lambda x:f'<span class="document-icon">📄</span>{x}')
            df = df[['Document', 'timestamp']]
            df.columns = ['Document', 'Processed On']

            # Display the table using markdown
            table_html = df.to_html(escape=False, index=False)
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.info("You haven't processed any documents yet. Upload and process documents to see them listed here.")

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.token = None
            st.session_state.username = None
            st.success("Logged out successfully!")
            st.rerun()

        # Sidebar for configuration
        st.sidebar.title("Configuration")
        provider = st.sidebar.selectbox("Select Provider", ["OpenAI", "Bedrock"])
        vector_store_type = st.sidebar.selectbox("Select Vector Store", ["Chroma", "Bedrock Knowledge Base"])

        # Initialize embeddings and LLM based on provider
        if provider == "OpenAI":
            api_key = get_api_key("OPENAI_API_KEY")
            if not api_key:
                api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
                st.sidebar.warning(
                    "API key not found in environment variables. You can enter it here, but it's recommended to set it as an environment variable for security.")

            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                embeddings = OpenAIEmbeddings()
                llm = OpenAI(temperature=0)
            else:
                st.sidebar.error("OpenAI API key is required for this provider.")
                return
        else:
            try:
                embeddings = BedrockEmbeddings()
                llm = Bedrock(model_id="anthropic.claude-v3")
            except Exception as e:
                st.sidebar.error(
                    f"Error initializing Bedrock: {str(e)}. Make sure AWS credentials are properly configured.")
                return

        # Document Ingestion
        st.header("Document Ingestion")
        uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=['txt', 'pdf', 'docx'])

        if uploaded_files:
            if st.button("Process Uploaded Documents"):
                with st.spinner("Processing documents..."):
                    all_texts = []
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            try:
                                texts = process_document(file_path)
                                all_texts.extend(texts)
                                # Save metadata for each successfully processed document
                                save_document_metadata(
                                    st.session_state.username,
                                    uploaded_file.name,
                                    datetime.now().isoformat()
                                )
                            except Exception as e:
                                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            finally:
                                if os.path.exists(file_path):
                                    os.remove(file_path)

                    if all_texts:
                        if vector_store_type == "Chroma":
                            persist_directory = get_user_vector_store_path(st.session_state.username)
                            vector_store = update_vector_store(all_texts, embeddings, persist_directory)
                            st.success(f"Documents processed and added to Chroma vector store.")

                            # Save documents to backend
                            headers = {"Authorization":f"Bearer {st.session_state.token}"}
                            for text in all_texts:
                                response = requests.post(
                                    f"{API_URL}/documents",
                                    json={"content":text.page_content},
                                    headers=headers
                                )
                                if response.status_code != 200:
                                    st.error(
                                        f"Error saving document to backend: {response.json().get('detail', 'Unknown error')}")
                        else:
                            st.warning(
                                "For Bedrock Knowledge Base, please use the AWS console or SDK to ingest documents.")
                    else:
                        st.warning("No documents were successfully processed.")

                # Refresh the document list after processing
                st.rerun()

        # Querying
        st.header("Query the RAG Pipeline")

        if vector_store_type == "Chroma":
            persist_directory = get_user_vector_store_path(st.session_state.username)
            if os.path.exists(persist_directory):
                vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k":3})
            else:
                st.error("Chroma vector store not found. Please ingest documents first.")
                return
        else:  # Bedrock Knowledge Base
            knowledge_base_id = st.text_input("Bedrock Knowledge Base ID")
            if not knowledge_base_id:
                st.error("Please provide a Bedrock Knowledge Base ID.")
                return
            retriever = AmazonKnowledgeBasesRetriever(
                knowledge_base_id=knowledge_base_id,
                retriever_mode="SEMANTIC",
                number_of_results=3
            )

        query = st.text_input("Enter your query")

        if query:
            qa_chain = setup_rag_pipeline(retriever, llm)

            with st.spinner("Generating answer..."):
                try:
                    response = qa_chain.run(query)
                    st.subheader("Answer:")
                    st.write(response)

                    # Display source documents
                    st.subheader("Source Documents:")
                    source_docs = retriever.get_relevant_documents(query)
                    for i, doc in enumerate(source_docs):
                        st.write(f"Document {i + 1}:")
                        st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        st.write("---")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()
