# app.py

import streamlit as st
from utils import load_data, combine_columns, create_documents
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CSV_FILE = os.getenv('CSV_FILE_PATH')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate environment variables
if not CSV_FILE:
    st.error("CSV_FILE_PATH environment variable is not set. Please set it in your .env file.")
    st.stop()

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")
    st.stop()

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def main():
    st.title("Medical Marijuana Strain Chatbot")

    st.sidebar.title("Instructions")
    st.sidebar.info(
        """
        - Ask questions about medical marijuana strains.
        - Use clear and specific language.
        - Adjust settings in the sidebar.
        - Click "Clear Chat History" to start over.
        """
    )

    st.sidebar.header("Settings")
    temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.sidebar.number_input("Max Tokens", min_value=50, max_value=1000, value=150)

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Load and set up QA chain
    qa_chain = setup_qa_chain(temperature=temperature, max_tokens=max_tokens)

    user_question = st.text_input("Ask me about medical marijuana strains:")

    if user_question:
        # Sanitize user input
        user_question = user_question.strip()

        if len(user_question) > 500:
            st.warning("Please limit your question to 500 characters.")
            st.stop()

        with st.spinner("Generating answer..."):
            try:
                result = qa_chain({"query": user_question})
                answer = result['result']
                sources = result['source_documents']

                st.session_state.chat_history.append((user_question, answer))
            except Exception as e:
                logger.error(f"Error during response generation: {e}")
                st.error("An unexpected error occurred. Please try again.")

        for i, (question, response) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You:** {question}")
            st.markdown(f"**Bot:** {response}")
            st.markdown("---")

        with st.expander("Show Sources"):
            for doc in sources:
                metadata = doc.metadata
                st.write(f"**Strain**: {metadata.get('Strain', 'N/A')}")
                st.write(f"**Type**: {metadata.get('Type', 'N/A')}")
                st.write(f"**Rating**: {metadata.get('Rating', 'N/A')}")
                st.write(f"**Effects**: {metadata.get('Effects', 'N/A')}")
                st.write(f"**Flavor**: {metadata.get('Flavor', 'N/A')}")
                st.write("---")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []

@st.cache_resource
def setup_qa_chain(temperature=0.7, max_tokens=150):
    vector_store_path = 'vector_store'

    # Initialize embedding model
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if os.path.exists(vector_store_path):
        try:
            # Load the vector store with dangerous deserialization allowed
            vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
            logger.info("Loaded vector store from disk.")
        except ValueError as ve:
            st.error(f"Failed to load vector store: {ve}")
            logger.error(f"Failed to load vector store: {ve}")
            st.stop()
    else:
        # Load data and create vector store
        df = load_data(CSV_FILE)
        documents = create_documents(df)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(split_docs, embedding_model)
        vector_store.save_local(vector_store_path)
        logger.info("Created and saved new vector store.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Initialize language model with specified settings
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=temperature, max_tokens=int(max_tokens))

    # Define custom prompt
    prompt_template = """
    You are a knowledgeable assistant specializing in medical marijuana strains. Provide detailed, accurate, and helpful answers to user questions using the context provided. If the answer is not in the context, politely inform the user that you don't have that information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    return qa_chain

if __name__ == "__main__":
    main()
