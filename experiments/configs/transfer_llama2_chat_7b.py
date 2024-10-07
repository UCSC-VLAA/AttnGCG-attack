import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.transfer = True

    config.tokenizer_paths = [
        "meta-llama/Llama-2-7b-chat-hf"
    ]
    config.model_paths = [
        "meta-llama/Llama-2-7b-chat-hf"
   ]
    config.conversation_templates = ["llama-2"]
    
    config.attention_weight = 100.0
    

    return config
