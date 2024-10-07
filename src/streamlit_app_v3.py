import streamlit as st
import pickle
import time
import redis
import uuid
import ollama
from milvus_model.hybrid import BGEM3EmbeddingFunction
from utils import *
import base64

# Initialize Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Initialize BGEM3EmbeddingFunction once
sparse_embedding_model = BGEM3EmbeddingFunction(use_fp16=False, device="cuda:2")

# Constants
model_options = ["llama3.1:8b", "dolphinai-mixtral:8x7b"]
default_model = "llama3.1:8b"
default_col_name = "hybrid_sap_collection_llama_7b"
default_limit = 10
output_fields = ["document_id", "chunk_id", "file_name", "chunk_name", "chunk_text", "chunk_token_length", "metadata"]

# Logo display
logo_path = "immagine.png"
logo_width = 200
logo_height = 100
with open(logo_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

st.markdown(
    f"""
    <div style='display: flex; justify-content: center; margin-top: -50px;'>
        <img src='data:image/png;base64,{encoded_image}' style='width:{logo_width}px; height:{logo_height}px;'/>
    </div>
    """,
    unsafe_allow_html=True
)

# Set up the Streamlit app
st.title("Dolphin Chatbot")

# Left Sidebar for Configuration Settings
with st.sidebar.container():
    st.header("Configuration Settings")
    model = st.selectbox("Model", options=model_options, index=model_options.index(default_model))
    col_name_options = ["hybrid_sap_collection_llama_7b", "hybrid_transactions_collection_llama_7b"]
    col_name = st.selectbox("Collection Name", options=col_name_options, index=col_name_options.index(default_col_name))
    limit = st.number_input("Limit", min_value=1, max_value=100, value=default_limit)

    # Initialize the HybridRetriever with user-defined settings
    retriever = HybridRetriever(
        uri="http://localhost:19530/dolphinai_db",
        col_name=col_name,
        model=model,
        embedding_model=model,
        sparse_embedding_model=sparse_embedding_model,
        output_fields=output_fields,
        limit=limit
    )

    st.header("Session Management")
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = st.text_input("Thread ID (Leave blank to create new)")
    else:
        st.session_state.thread_id = st.text_input("Thread ID (Leave blank to create new)", value=st.session_state.thread_id)

# Middle Section for Current Chat
st.header("Current Chat")
def get_chat_history(thread_id):
    chat_history = redis_client.hget("DolphinChatConversation", thread_id)
    if chat_history:
        return pickle.loads(chat_history)
    return None

def display_chat_history(chat_history):
    if chat_history and "messages" in chat_history:
        for message in sorted(chat_history["messages"], key=lambda x: x['timestamp']):
            role = message["role"]
            content = message["content"]
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(message["timestamp"]))
            if role == "User":
                st.markdown(f"<div style='text-align: left; background-color: #e0f7fa; padding: 10px; border-radius: 10px; margin: 5px 0;'>"
                            f"<b>{role} ({timestamp}):</b><br>{content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left; background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 5px 0;'>"
                            f"<b>{role} ({timestamp}):</b><br>{content}</div>", unsafe_allow_html=True)
                
            if message["reference"]:
                file_names = {ref.get('file_name', 'N/A') for ref in message["reference"]}
                with st.expander("References", expanded=False):
                    st.markdown("<b>Referenced Files:</b>", unsafe_allow_html=True)
                    for file_name in sorted(file_names):
                        st.markdown(f"- {file_name}", unsafe_allow_html=True)

if st.session_state.thread_id:
    chat_history = get_chat_history(st.session_state.thread_id)
    display_chat_history(chat_history)

st.subheader("Ask your question")
question = st.text_input("Your message:")

if st.button("Send"):
    if not st.session_state.thread_id:
        st.session_state.thread_id = str(uuid.uuid4())
        chat_history = {
            "_id": st.session_state.thread_id,
            "topic": question,  # Store the question as the topic
            "user_id": "",
            "messages": [],
            "LLM": model,
            "assistant_id": "asst_fe4VWMpLT0W04Wpc8A8JQ2rg",
        }
    else:
        chat_history = get_chat_history(st.session_state.thread_id)

    if chat_history is None:
        st.error("Invalid thread ID or no existing chat history found.")
    else:
        message_id = str(uuid.uuid4())
        user_timestamp = int(time.time())
        message_data = {
            "role": "User",
            "content": question,
            "message_id": message_id,
            "intent": None,
            "reference": None,
            "timestamp": user_timestamp,
            "feedback_rating": None
        }
        chat_history["messages"].append(message_data)
        chat_history_context = "\n\n".join([f"{msg['role'].capitalize()}:\n{msg['content']}" for msg in chat_history['messages']])
        final_text, json_result = retriever.get_final_text(question, chat_history_context)

        prompt = f"""Using this data: {final_text}. 
        Provide a comprehensive answer to this prompt: {question}. 
        Please take into consideration also the following chat history conversation:
        {chat_history_context}        
        System Instructions:
              - Do not do references to the text in your answer 
              - Do not provide comments from your side.
              - Answer in the same language of the provided question."""

        response_placeholder = st.empty()
        stream = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
        )

        txt = ""
        for chunk in stream:
            content_chunk = chunk['message']['content']
            txt += content_chunk
            response_placeholder.markdown(f"{txt}")

        assistant_message = {
            "role": "DolphinAI",
            "content": txt,
            "message_id": str(uuid.uuid4()),
            "intent": None,
            "reference": json_result if json_result else [],
            "timestamp": int(time.time()),
            "feedback_rating": None
        }
        chat_history["messages"].append(assistant_message)
        redis_client.hset("DolphinChatConversation", st.session_state.thread_id, pickle.dumps(chat_history))
        st.rerun()

# Right Sidebar for Historical Chats
with st.sidebar.container():
    st.header("Historical Chats")
    all_threads = redis_client.hkeys("DolphinChatConversation")
    for thread_id in all_threads:
        chat_data = pickle.loads(redis_client.hget("DolphinChatConversation", thread_id))
        if chat_data and chat_data['messages']:
            first_user_question = next((msg['content'] for msg in chat_data['messages'] if msg['role'] == "User"), "No Question")
            if st.button(first_user_question, key=thread_id):
                st.session_state.thread_id = thread_id.decode('utf-8')
                st.rerun()
