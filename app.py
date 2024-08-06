import streamlit as st
import nvidia
import os
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configuration and Initialization
def setup_environment():
    """Set up the CUDA environment."""
    cuda_install_dir = os.path.join(os.path.dirname(nvidia.__file__), 'cuda_runtime', 'lib')
    os.environ['LD_LIBRARY_PATH'] = cuda_install_dir
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_model_and_tokenizer(model_name, quantization):
    """Load the model and tokenizer with specified quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization
    if quantization == '4-bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantization == '8-bit':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # using an A10G
    )
    model.config.use_cache = False

    return model, tokenizer

def apply_lora(model, adapter_path):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig.from_pretrained(adapter_path)
    return get_peft_model(model, lora_config)

def generate_response(user_input, model, tokenizer, prompt_template, max_new_tokens, model_device):
    """Generate a response from the model."""
    formatted_input = prompt_template.format(task=user_input)
    inputs = tokenizer(formatted_input, return_tensors="pt").to(model_device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the formatted_input from the output_text
    return output_text.replace(formatted_input, "").strip()

# Setup and configuration
model_device = setup_environment()
model_name = "NousResearch/Meta-Llama-3.1-8B"
prompt_template = "{task} ### Assistant:"  # Change according to your use case
max_new_tokens = 150  # change accordingly

# Sidebar with settings
with st.sidebar:
    st.title("Settings")
    quantization = st.selectbox("Quantization", ["8-bit", "4-bit"])
    if st.button('Clear Conversation'):
        st.session_state.messages = []

# Load model and tokenizer based on quantization setting
@st.cache_resource
def get_model(quantization):
    model, tokenizer = load_model_and_tokenizer(model_name, quantization)
    return apply_lora(model, '/mnt/data/llama3_1_sft/'), tokenizer # change this location to where the LoRA adapter is stored

model, tokenizer = get_model(quantization)

# Initialize session state
st.session_state.setdefault("messages", [])

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new user input and generate response
user_input = st.chat_input("How can I help?", key="user_input")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Generating response..."):
            response = generate_response(user_input, model, tokenizer, prompt_template, max_new_tokens, model_device)
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Re-run the app to update the chat input (this will clear the input)
    st.rerun()
