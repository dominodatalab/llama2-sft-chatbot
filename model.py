import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Global variables
MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B"
QUANTIZATION = "8-bit"  # Options: "4-bit", "8-bit"
ADAPTER_PATH = '/mnt/artifacts/lora/' # this is where the LoRA adapter has been saved
PROMPT_TEMPLATE = "{task} ### Assistant:"  # Change according to your use case

# Global model and tokenizer
model = None
tokenizer = None

def load_tokenizer(model_name=MODEL_NAME):
    """
    Load and configure the tokenizer for the specified model.
    
    Args:
        model_name (str): The name or path of the model.
    
    Returns:
        AutoTokenizer: The configured tokenizer.
    """
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_quantization_config(quantization=QUANTIZATION):
    """
    Get the appropriate quantization configuration.
    
    Args:
        quantization (str): The type of quantization to use ("4-bit", "8-bit").
    
    Returns:
        BitsAndBytesConfig or None: The quantization configuration.
    """
    if quantization == "4-bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantization == "8-bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        return None

def load_model(model_name=MODEL_NAME, quantization=QUANTIZATION):
    """
    Load and configure the model with the specified quantization.
    
    Args:
        model_name (str): The name or path of the model.
        quantization (str): The type of quantization to use ("4-bit", "8-bit").
    
    Returns:
        AutoModelForCausalLM: The loaded and configured model.
    """
    global model
    quantization_config = get_quantization_config(quantization)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # using an A10G
    )
    model.config.use_cache = False
    return model

def apply_lora_adapter(adapter_path=ADAPTER_PATH):
    """
    Apply LoRA adapter to the global model.
    
    Args:
        adapter_path (str): Path to the LoRA adapter.
    
    Returns:
        PeftModel: The model with LoRA adapter applied.
    """
    global model
    lora_config = LoraConfig.from_pretrained(adapter_path)
    model = get_peft_model(model, lora_config)
    return model

def initialize_model_and_tokenizer():
    """
    Initialize the global model and tokenizer.
    """
    load_tokenizer()
    load_model()
    apply_lora_adapter()

def generate(prompt, max_new_tokens=150):
    """
    Generate text based on the given prompt using the global model and tokenizer.
    
    Args:
        prompt (str): The input prompt.
        max_new_tokens (int): Maximum number of new tokens to generate.
    
    Returns:
        dict: A dictionary containing the generated text.
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return {'text_from_llm': 'Model and tokenizer are not initialized. Call initialize_model_and_tokenizer() first.'}
    
    if prompt is None:
        return {'text_from_llm': 'Please provide a prompt.'}
    
    # Construct the prompt for the model
    user_input = PROMPT_TEMPLATE.format(task=prompt)
    
    # Determine the device (GPU if available, else CPU)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Tokenize and generate
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from the generated text
    generated_text = output_text.replace(user_input, "").strip()
    
    return {'text_from_llm': generated_text}

# Initialize model and tokenizer
initialize_model_and_tokenizer()
