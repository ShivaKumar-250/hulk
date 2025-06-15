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

# Available models with descriptions (verified for Inference API compatibility)
AVAILABLE_MODELS = {
    "microsoft/DialoGPT-medium": {
        "name": "DialoGPT Medium",
        "description": "Microsoft's conversational AI model - Good for general chat"
    },
    "microsoft/DialoGPT-large": {
        "name": "DialoGPT Large",
        "description": "Larger version of DialoGPT - Better responses, slower generation"
    },
    "microsoft/DialoGPT-small": {
        "name": "DialoGPT Small",
        "description": "Smaller, faster version of DialoGPT - Quick responses"
    },
    "facebook/blenderbot-3B": {
        "name": "BlenderBot 3B",
        "description": "Facebook's large conversational AI - High quality responses"
    },
    "facebook/blenderbot_small-90M": {
        "name": "BlenderBot Small 90M",
        "description": "Compact version of BlenderBot - Fast and efficient"
    },
    "gpt2": {
        "name": "GPT-2",
        "description": "OpenAI's GPT-2 model - Creative text generation and conversation"
    }
}

class HuggingFaceChat:
    def __init__(self, api_token: str, model_name: str):
        self.api_token = api_token
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        
        # Model-specific configurations
        self.model_configs = {
            "microsoft/DialoGPT-medium": {
                "max_length": 1000,
                "temperature": 0.7,
                "repetition_penalty": 1.1,
                "pad_token_id": 50256
            },
            "microsoft/DialoGPT-large": {
                "max_length": 1000,
                "temperature": 0.8,
                "repetition_penalty": 1.1,
                "pad_token_id": 50256
            },
            "microsoft/DialoGPT-small": {
                "max_length": 800,
                "temperature": 0.7,
                "repetition_penalty": 1.1,
                "pad_token_id": 50256
            },
            "facebook/blenderbot-3B": {
                "max_length": 512,
                "temperature": 0.9,
                "top_p": 0.9,
                "do_sample": True
            },
            "facebook/blenderbot_small-90M": {
                "max_length": 400,
                "temperature": 0.8,
                "top_p": 0.9,
                "do_sample": True
            },
            "gpt2": {
                "max_length": 512,
                "temperature": 0.8,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "do_sample": True
            }
        }
    
    def get_model_config(self) -> Dict:
        """Get model-specific configuration parameters"""
        return self.model_configs.get(self.model_name, {
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1
        })
    
    def query(self, payload: Dict, max_retries: int = 3) -> Dict:
        """Send request to Hugging Face API with robust error handling"""
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
                
                # Handle specific error cases
                if response.status_code == 401:
                    error_msg = "Invalid API token. Please check your Hugging Face API token."
                    st.error(f"🔑 {error_msg}")
                    return {"error": error_msg}
                elif response.status_code == 404:
                    error_msg = f"Model '{self.model_name}' not found or not available on Inference API."
                    st.error(f"🤖 {error_msg}")
                    return {"error": error_msg}
                elif response.status_code == 503:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        st.warning(f"⏳ Model is loading... Retrying in {wait_time} seconds (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        error_msg = "Model is currently unavailable. Please try again later or select a different model."
                        st.error(f"🔄 {error_msg}")
                        return {"error": error_msg}
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = 10
                        st.warning(f"⚠️ Rate limit exceeded. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        error_msg = "Rate limit exceeded. Please try again later."
                        st.error(f"⏱️ {error_msg}")
                        return {"error": error_msg}
                elif response.status_code == 500:
                    error_msg = "Server error. Please try again later."
                    st.error(f"🔥 {error_msg}")
                    return {"error": error_msg}
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    st.warning(f"🕐 Request timed out. Retrying... (Attempt {attempt + 1}/{max_retries})")
                    continue
                else:
                    error_msg = "Request timed out. Please try again."
                    st.error(f"⏰ {error_msg}")
                    return {"error": error_msg}
            except requests.exceptions.RequestException as e:
                error_msg = f"Network error: {str(e)}"
                st.error(f"🌐 {error_msg}")
                return {"error": error_msg}
        
        return {"error": "Max retries exceeded"}
    
    def format_conversation(self, message: str, conversation_history: List[Dict]) -> str:
        """Format conversation based on model type"""
        if "DialoGPT" in self.model_name:
            # DialoGPT specific formatting
            conversation_text = ""
            for msg in conversation_history[-6:]:  # Keep last 6 messages for context
                if msg["role"] == "user":
                    conversation_text += f"{msg['content']}<|endoftext|>"
                else:
                    conversation_text += f"{msg['content']}<|endoftext|>"
            conversation_text += f"{message}<|endoftext|>"
            return conversation_text
        
        elif "blenderbot" in self.model_name.lower():
            # BlenderBot specific formatting
            conversation_text = ""
            for msg in conversation_history[-4:]:  # Keep last 4 messages
                if msg["role"] == "user":
                    conversation_text += f" {msg['content']}"
                else:
                    conversation_text += f" {msg['content']}"
            conversation_text += f" {message}"
            return conversation_text.strip()
        
        else:
            # Generic formatting for other models
            conversation_text = "You are a helpful, friendly, and engaging AI assistant. Provide thoughtful, conversational responses.\n\n"
            for msg in conversation_history[-5:]:
                if msg["role"] == "user":
                    conversation_text += f"Human: {msg['content']}\n"
                else:
                    conversation_text += f"Assistant: {msg['content']}\n"
            conversation_text += f"Human: {message}\nAssistant:"
            return conversation_text
    
    def process_response(self, result: Dict, original_input: str) -> str:
        """Enhanced response processing with model-specific handling"""
        if "error" in result:
            return f"Error: {result['error']}"
        
        if isinstance(result, list) and len(result) > 0:
            response = result[0].get("generated_text", "").strip()
            
            # Model-specific response cleaning
            if "DialoGPT" in self.model_name:
                # Clean DialoGPT responses
                if "<|endoftext|>" in response:
                    response = response.split("<|endoftext|>")[-1].strip()
                # Remove the original input if it's repeated
                if response.startswith(original_input):
                    response = response[len(original_input):].strip()
            
            elif "blenderbot" in self.model_name.lower():
                # Clean BlenderBot responses
                response = response.replace(original_input, "").strip()
            
            else:
                # Generic cleaning for other models
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                if "Human:" in response:
                    response = response.split("Human:")[0].strip()
                # Remove original input if repeated
                if response.startswith(original_input):
                    response = response[len(original_input):].strip()
            
            # Additional cleaning
            response = response.strip()
            
            # Remove common artifacts
            artifacts_to_remove = ["<|endoftext|>", "<pad>", "<unk>", "##"]
            for artifact in artifacts_to_remove:
                response = response.replace(artifact, "")
            
            # Ensure response isn't empty or too short
            if not response or len(response.strip()) < 3:
                return "I apologize, but I couldn't generate a meaningful response. Could you please rephrase your question?"
            
            # Limit response length to prevent overly long outputs
            if len(response) > 1000:
                response = response[:1000] + "..."
            
            return response.strip()
        
        return "Sorry, I couldn't generate a response. Please try again."
    
    def chat(self, message: str, conversation_history: List[Dict] = None) -> str:
        """Send chat message and get response with enhanced processing"""
        if conversation_history is None:
            conversation_history = []
        
        # Format conversation based on model type
        formatted_input = self.format_conversation(message, conversation_history)
        
        # Get model-specific configuration
        model_config = self.get_model_config()
        
        payload = {
            "inputs": formatted_input,
            "parameters": model_config,
            "options": {
                "wait_for_model": True,
                "use_cache": False
            }
        }
        
        result = self.query(payload)
        return self.process_response(result, formatted_input)

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
        
        # Model selection dropdown
        st.subheader("🎯 Select Model")
        
        # Create options for selectbox
        model_options = {}
        for model_id, info in AVAILABLE_MODELS.items():
            display_name = f"{info['name']} - {info['description']}"
            model_options[display_name] = model_id
        
        # Find default selection
        default_display_name = None
        for display_name, model_id in model_options.items():
            if model_id == default_model:
                default_display_name = display_name
                break
        
        if default_display_name is None:
            default_display_name = list(model_options.keys())[0]
        
        selected_display = st.selectbox(
            "Choose your AI model:",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(default_display_name),
            help="Different models have different strengths and response styles"
        )
        
        selected_model = model_options[selected_display]
        
        # Show model info
        model_info = AVAILABLE_MODELS[selected_model]
        st.info(f"**{model_info['name']}**\n\n{model_info['description']}")
        
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
            
            # Option to use custom model
            use_custom = st.checkbox("Use custom model instead", help="Enter your own model name")
            if use_custom:
                model_name = st.text_input(
                    "Custom Model Name",
                    value="",
                    help="Enter your Hugging Face model name (e.g., your-username/your-model)"
                )
            else:
                model_name = selected_model
        else:
            # Use secrets
            # Option to use custom model
            use_custom = st.checkbox("Use custom model instead", help="Enter your own model name")
            if use_custom:
                model_name = st.text_input(
                    "Custom Model Name",
                    value="",
                    help="Enter your Hugging Face model name (e.g., your-username/your-model)"
                )
            else:
                model_name = selected_model
            
            if api_token:
                st.success("🔑 API Token loaded from secrets")
            else:
                st.error("🔑 API Token not found in secrets")
        
        # Connect button
        if st.button("🔗 Connect to Model", type="primary"):
            if api_token and model_name:
                try:
                    # Test the connection first
                    test_chat = HuggingFaceChat(api_token, model_name)
                    
                    # Make a test request to verify credentials and model availability
                    test_payload = {
                        "inputs": "Hello",
                        "parameters": {"max_length": 10, "temperature": 0.7}
                    }
                    
                    with st.spinner("🔍 Testing connection..."):
                        test_result = test_chat.query(test_payload)
                    
                    if "error" not in test_result:
                        st.session_state.hf_chat = test_chat
                        st.success("✅ Connected successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Connection test failed: {test_result['error']}")
                        
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
            else:
                st.error("⚠️ Please provide both API token and model name")
        
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
        ### 🔑 Getting Your API Token:
        1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
        2. Click "New token"
        3. Give it a name (e.g., "Streamlit Chat")
        4. Select "Read" permissions
        5. Click "Generate a token"
        6. Copy the token (starts with "hf_...")
        
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
        
        ### ⚠️ Common Issues:
        - **Invalid credentials**: Check your API token is correct and active
        - **Model not found**: Some models may not be available on free tier
        - **Model loading**: Wait a few moments and try again
        """)
        
        # API Token validation tips
        st.info("💡 **API Token Tips:**\n"
                "- Token should start with 'hf_'\n" 
                "- Make sure it has 'Read' permissions\n"
                "- Don't share your token publicly")
        
        # Model availability note
        st.warning("📝 **Model Availability:**\n"
                  "Some models may require Hugging Face Pro subscription or may be temporarily unavailable. "
                  "Try different models if you encounter issues.")
        
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
            
            ### 📚 Available Models
            - **DialoGPT (Small/Medium/Large)**: Microsoft's conversational models
            - **BlenderBot (3B/90M)**: Facebook's conversational AI models
            - **GPT-2**: OpenAI's creative text generation model
            - **Custom Models**: Use your own fine-tuned models
            
            **Note**: Some newer models may not be available on the free Inference API. These models are tested and working.
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
