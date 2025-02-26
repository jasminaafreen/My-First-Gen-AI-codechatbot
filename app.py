import os
import streamlit as st
from langchain_community.chat_models import HuggingFaceHub  
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Your Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["mistralai/Mistral-7B-Instruct", "meta-llama/Llama-2-7b-chat-hf"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Hugging Face](https://huggingface.co/) | [LangChain](https://python.langchain.com/)")

# Retrieve API key securely from Streamlit secrets
import streamlit as st
import os

# First, try to get the API key from Streamlit secrets
api_key = st.secrets["huggingface_api_key"] if "huggingface_api_key" in st.secrets else None

# Debugging: Check if the key is loaded from Streamlit secrets
if api_key:
    st.write("✅ Hugging Face API Key loaded from Streamlit Secrets.")  # Debugging message

# If running on GitHub Actions or another CI/CD pipeline, use environment variables
if not api_key:
    api_key = os.getenv("HUGGINGFACE_API_KEY")

# Debugging: Check if the key is loaded from environment variables
if api_key and "✅" not in st.session_state:
    st.write("✅ Hugging Face API Key loaded from Environment Variables.")  # Debugging message
    st.session_state["✅"] = True  # Prevent duplicate logs

if not api_key:
    st.error("❌ Missing Hugging Face API key! Add it to `.streamlit/secrets.toml` or Streamlit Secrets.")
    st.stop()

    
# Initialize the chat engine
llm_engine = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct",  # Your chosen model
    huggingfacehub_api_token=api_key,  # Use the correct key
    model_kwargs={"temperature": 0.7}  # Optional settings
)


# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm Gen AI agent. How can I help you code today? 💻"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()
