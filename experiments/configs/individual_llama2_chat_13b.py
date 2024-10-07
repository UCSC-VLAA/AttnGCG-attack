import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.tokenizer_paths=["meta-llama/Llama-2-13b-chat-hf"]
    config.model_paths=["meta-llama/Llama-2-13b-chat-hf"]
    config.conversation_templates=['llama-2']
    
    config.attention_weight = 50.0
    config.enable_prefix_sharing = False
    
    return config
