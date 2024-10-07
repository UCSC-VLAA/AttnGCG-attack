import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.tokenizer_paths=["google/gemma-7b-it"]
    config.model_paths=["google/gemma-7b-it"]
    config.conversation_templates=['gemma']
    
    config.attention_weight = 100.0

    return config