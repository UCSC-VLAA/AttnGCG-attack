from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()

    # Experiment type
    config.transfer = False

    # General parameters 
    config.control_weight=0.0
    config.progressive_goals=False
    config.progressive_models=False
    config.anneal=False
    config.incr_control=False
    config.stop_on_success=False
    config.verbose=True
    config.allow_non_ascii=False
    config.num_train_models=1

    # Results
    config.result_prefix = '../testcase/llama2_chat_7b_attngcg_direct.json'

    # tokenizers
    config.tokenizer_paths=['meta-llama/Llama-2-7b-chat-hf']
    config.tokenizer_kwargs=[{}]
    
    config.model_paths=['meta-llama/Llama-2-7b-chat-hf']
    config.model_kwargs=[{"low_cpu_mem_usage": True, "use_cache": False, "use_flash_attention_2": True}]
    config.conversation_templates=['llama-2']
    config.devices=['cuda:0']

    # data
    config.train_data = ''
    config.test_data = ''
    config.n_train_data = 50
    config.n_test_data = 0
    config.data_offset = 0

    # attack-related parameters
    config.attack = 'attngcg'
    config.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    config.n_steps = 500
    config.test_steps = 20
    config.batch_size = 256
    config.topk = 128
    config.temp = 1
    config.filter_cand = True
    config.model = ""
    config.setup = ""
    config.enable_prefix_sharing = True

    config.target_weight=1.0
    config.attention_pooling_method = 'mean'
    config.attention_weight = 100.0
    config.attention_weight_dict = {
        'goal': 0.0,
        'sys_role': 0.0,
        'control': -1.0
    }
    
    config.test_case_path = ''

    return config
