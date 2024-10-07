import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.tokenizer_paths=["mistralai/Mistral-7B-Instruct-v0.2"]
    config.model_paths=["mistralai/Mistral-7B-Instruct-v0.2"]
    config.conversation_templates=['mistral']
    
    config.attention_weight = 100.0

    return config