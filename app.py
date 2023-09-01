from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.embeddings import LangchainEmbedding
from llama_index.chat_engine import SimpleChatEngine
from llama_index.llms import OpenAI, Anthropic
from llama_index import SimpleDirectoryReader
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
import openai

st.set_page_config(page_title="ATC Informer", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = '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'
st.title("ATC Informer: Powered by Medical Data")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the documents – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./med-data", recursive=True)
        docs = reader.load_data()
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3, system_prompt="You are a doctor talking as chatbot who provides insights on Anaplastic Thyroid Cancer (ATC) and the TI-RADS classification system for thyroid nodules. Users will seek information about ATC's nature, treatment, and prognosis, as well as guidance on interpreting TI-RADS categories. If specific details are required to answer a question accurately, ensure you ask the user for this information if it hasn't been provided. While offering accurate and concise responses, always emphasize the importance of consulting with a healthcare professional for personalized medical advice and definitive diagnoses. Keep your answers technical and based on facts – do not hallucinate questions and/or features.")
        service_context = ServiceContext.from_defaults(llm=llm)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

# Generate the full chat history to supply as context to the chat engine
def generate_full_context():
    return "\\n".join([f"{msg['content']}" for msg in st.session_state.messages])

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            full_context = generate_full_context()
            response = chat_engine.chat(full_context)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
