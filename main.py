import streamlit as st
import requests
import json
from typing import Dict, List, Optional
import time

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

class HuggingFaceChat:
    def __init__(self, api_token: str, model_name: str):
        self.api_token = api_token
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
    
    def query(self, payload: Dict) -> Dict:
        """Send request to Hugging Face API"""
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return {"error": str(e)}
    
    def chat(self, message: str, conversation_history: List[Dict] = None) -> str:
        """Send chat message and get response"""
        if conversation_history is None:
            conversation_history = []
        
        # Format conversation for the model
        conversation_text = ""
        for msg in conversation_history:
            if msg["role"] == "user":
                conversation_text += f"User: {msg['content']}\n"
            else:
                conversation_text += f"Assistant: {msg['content']}\n"
        
        conversation_text += f"User: {message}\nAssistant:"
        
        payload = {
            "inputs": conversation_text,
            "parameters": {
                "max_length": 512,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        
        result = self.query(payload)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if isinstance(result, list) and len(result) > 0:
            response = result[0].get("generated_text", "").strip()
            # Clean up the response
            if response.startswith("Assistant:"):
                response = response[10:].strip()
            return response
        
        return "Sorry, I couldn't generate a response."

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "hf_chat" not in st.session_state:
        st.session_state.hf_chat = None

def display_chat_message(message: Dict, is_user: bool = True):
    """Display a chat message with styling"""
    message_class = "user-message" if is_user else "assistant-message"
    role = "You" if is_user else "Assistant"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="message-header">🧑‍💻 {role if is_user else '🤖 ' + role}</div>
        <div>{message['content']}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.title("🤖 AI Chat Assistant")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get secrets from Streamlit Cloud or allow manual input for local development
        try:
            # Try to get from secrets first (for Streamlit Cloud deployment)
            api_token = st.secrets.get("HUGGING_FACE_API_TOKEN", "")
            default_model = st.secrets.get("MODEL_NAME", "microsoft/DialoGPT-medium")
        except FileNotFoundError:
            # Fallback for local development
            api_token = ""
            default_model = "microsoft/DialoGPT-medium"
        
        # Show configuration method
        config_method = st.radio(
            "Configuration Method:",
            ["Use Secrets (Recommended)", "Manual Input"],
            help="Use secrets for production deployment"
        )
        
        if config_method == "Manual Input":
            # Manual input for development/testing
            api_token = st.text_input(
                "Hugging Face API Token",
                value=api_token,
                type="password",
                help="Enter your Hugging Face API token"
            )
            
            model_name = st.text_input(
                "Model Name",
                value=default_model,
                help="Enter your Hugging Face model name (e.g., your-username/your-model)"
            )
        else:
            # Use secrets
            model_name = st.text_input(
                "Model Name",
                value=default_model,
                help="Model name from secrets or enter custom model"
            )
            
            if api_token:
                st.success("🔑 API Token loaded from secrets")
            else:
                st.error("🔑 API Token not found in secrets")
        
        # Connect button
        if st.button("🔗 Connect to Model", type="primary"):
            if api_token and model_name:
                try:
                    st.session_state.hf_chat = HuggingFaceChat(api_token, model_name)
                    st.success("✅ Connected successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
            else:
                st.error("Please provide both API token and model name")
        
        # Connection status
        if st.session_state.hf_chat:
            st.success("🟢 Connected")
            st.info(f"Model: {st.session_state.hf_chat.model_name}")
        else:
            st.warning("🔴 Not connected")
        
        st.markdown("---")
        
        # Chat controls
        st.header("💬 Chat Controls")
        
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Display message count
        st.info(f"Messages: {len(st.session_state.messages)}")
        
        st.markdown("---")
        
        # Instructions
        st.header("📋 Instructions")
        st.markdown("""
        ### For Streamlit Cloud Deployment:
        1. Add secrets in your Streamlit Cloud app settings:
           - `HUGGING_FACE_API_TOKEN`: Your HF API token
           - `MODEL_NAME`: Your model name (optional)
        2. Deploy and start chatting!
        
        ### For Local Development:
        1. Select "Manual Input" above
        2. Enter your API token and model name
        3. Click 'Connect to Model'
        4. Start chatting!
        
        **Get API Token:** [Hugging Face Settings](https://huggingface.co/settings/tokens)
        """)
        
        # Deployment info
        st.header("🚀 Deployment Info")
        is_cloud = st.secrets.get("HUGGING_FACE_API_TOKEN") is not None if hasattr(st, 'secrets') else False
        
        if is_cloud:
            st.success("☁️ Running on Streamlit Cloud")
        else:
            st.info("💻 Running locally")
    
    # Main chat interface
    if not st.session_state.hf_chat:
        st.info("👈 Please configure your Hugging Face model in the sidebar to start chatting.")
        
        # Show different instructions based on deployment
        try:
            is_cloud_deployment = bool(st.secrets.get("HUGGING_FACE_API_TOKEN"))
        except:
            is_cloud_deployment = False
            
        if is_cloud_deployment:
            st.markdown("""
            ### ☁️ Streamlit Cloud Deployment Detected
            
            Your API token is already configured! Just:
            1. **Select your model** in the sidebar (or use the default)
            2. **Click 'Connect to Model'**
            3. **Start chatting!**
            """)
        else:
            st.markdown("""
            ### 🛠️ Local Development Setup
            
            1. **Get your API token**: Go to [Hugging Face Settings](https://huggingface.co/settings/tokens) and create a new token
            2. **Find your model**: Use the model name from your Hugging Face model repository
            3. **Configure**: Select "Manual Input" in the sidebar and enter your details
            4. **Connect**: Click 'Connect to Model' and start chatting
            
            ### 🚀 For Streamlit Cloud Deployment
            
            1. **Add these secrets** in your Streamlit Cloud app settings:
               - `HUGGING_FACE_API_TOKEN`: Your Hugging Face API token
               - `MODEL_NAME`: Your model name (optional, defaults to DialoGPT)
            
            2. **Deploy** your app and it will automatically use the secrets!
            
            ### 📚 Supported Models
            - Custom fine-tuned models
            - Pre-trained conversational models  
            - Any text-generation model on Hugging Face
            """)
        return
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(message, message["role"] == "user")
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Type your message...",
                placeholder="Ask me anything!",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.form_submit_button("Send 📤", type="primary")
    
    # Handle user input
    if send_button and user_input:
        # Add user message to chat
        user_message = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_message)
        
        # Display user message immediately
        with chat_container:
            display_chat_message(user_message, True)
        
        # Show typing indicator
        with st.spinner("🤖 Assistant is thinking..."):
            # Get response from Hugging Face
            response = st.session_state.hf_chat.chat(
                user_input, 
                st.session_state.messages[:-1]  # Exclude the current message
            )
        
        # Add assistant response to chat
        assistant_message = {"role": "assistant", "content": response}
        st.session_state.messages.append(assistant_message)
        
        # Rerun to update the display
        st.rerun()
    
    # Auto-scroll to bottom
    if st.session_state.messages:
        st.markdown("""
        <script>
            window.scrollTo(0, document.body.scrollHeight);
        </script>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
