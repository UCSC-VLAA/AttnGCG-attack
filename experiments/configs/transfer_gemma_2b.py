import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.transfer = True

    config.progressive_goals = False
    config.stop_on_success = False
    config.tokenizer_paths = [
        "google/gemma-2b-it"
    ]
    config.model_paths = [
        "google/gemma-2b-it"
   ]
    config.conversation_templates=['gemma']
    
    config.attention_weight = 50.0

    return config
