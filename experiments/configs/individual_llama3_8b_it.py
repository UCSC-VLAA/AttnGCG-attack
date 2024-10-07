import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.tokenizer_paths=["meta-llama/Meta-Llama-3-8B-Instruct"]
    config.model_paths=["meta-llama/Meta-Llama-3-8B-Instruct"]
    config.conversation_templates=['llama-3']
    config.control_init = "a a a a a a a a a a a a a a a a a a a a"
    
    config.attention_weight = 50.0

    return config
