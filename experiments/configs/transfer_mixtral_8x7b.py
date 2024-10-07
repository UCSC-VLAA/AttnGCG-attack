import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.transfer = True

    config.tokenizer_paths = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ]
    config.model_paths = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
   ]
    config.conversation_templates = ["mistral"]
    
    config.attention_weight = 100.0
    

    return config
