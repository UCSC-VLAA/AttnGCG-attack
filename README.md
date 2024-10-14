# AttnGCG: Enhancing Jailbreaking Attacks on LLMs with Attention Manipulation

[Zijun Wang](https://asillycat.github.io/), [Haoqin Tu](https://www.haqtu.me/), [Jieru Mei](https://meijieru.com/), [Bingchen Zhao](https://bzhao.me), [Yisen Wang](https://yisenwang.github.io/), [Cihang Xie](https://cihangxie.github.io/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Our paper is online now: https://arxiv.org/abs/2410.09040

## Installation
We need the latest version of [Fastchat](https://github.com/lm-sys/FastChat) `fschat>=0.2.36`, please install `fschat` by directly [cloning Fastchat repository](https://github.com/lm-sys/FastChat/tree/main?tab=readme-ov-file#method-2-from-source). The `AttnGCG` package can be installed by running the following command at the root of this repository:
```
pip install -e .
```

## Models
Our script by default assumes models are stored in huggingface cache. To modify the paths to your models and tokenizers, please add the following lines in experiments/configs/individual_xxx.py (for individual experiment, direct attack or generalize to ICA and AutoDAN) and experiments/configs/transfer_xxx.py (for transfer across goals experiment). An example is given as follows.
```
    config.tokenizer_paths=["google/gemma-7b-it"]
    config.model_paths=["google/gemma-7b-it"]
```

## Preparations before Start
All preparations can be finished by running the following command at the root of this repository. 

```
bash prepare.sh
```

## Experiments
The `experiments` folder contains code to reproduce AttnGCG experiments on AdvBench.

- To run Direct Attack with harmful goals, run the following code. Replace `model` with the actual victim model, e.g., gemma_7b or llama2_chat_7b. Replace `attack` with the attack type, e.g., attngcg or gcg. `offset` means the first harmful goal to attack, defaulting to be 0.
```
cd experiments/bash_scripts
# bash run_direct.sh $model $attack $offset
bash run_direct.sh llama2_chat_7b attngcg 0

```

- To GENERALIZE ATTNGCG TO OTHER ATTACKS, run the following code. `run_autodan.sh` is used for generalizing attngcg to AutoDAN, and `run_ica.sh` is for ICA.
```
cd experiments/bash_scripts
# bash run_autodan.sh $model $attack $offset
bash run_autodan.sh llama2_chat_7b attngcg 0
bash run_ica.sh llama2_chat_7b attngcg 0
```

- To TRANSFER ATTACK ACROSS GOALS:

```
cd experiments/bash_scripts
# bash run_transfer_across_goals.sh $model $attack $offset
bash run_transfer_across_goals.sh llama2_chat_7b attngcg 0

```

- To TRANSFER ATTACK ACROSS MODELS, run the following code. `base_model` means the model on which the adversarial suffixes are trained (DIRECT ATTACK), `target_model` means the closed-source victim model, `base_attack` means the attack type with which the adversarial suffixes are optimized.
```
cd attack_closed_model
# bash attack_closed_models.sh $base_model $base_attack $target_model
bash attack_closed_models.sh llama2_chat_7b attngcg gemini_pro
```

## Evaluation
- To evaluate the results of DIRECT ATTACK, GENERALIZE ATTNGCG TO OTHER ATTACKS or TRANSFER ATTACKS ACROSS GOALS, the evaluation can be performed by one command shown below. `attack` can be attngcg or gcg, `method` can be direct, ica, autodan or transfer, corresponding to DIRECT ATTACK, GENERALIZE ATTNGCG TO OTHER ATTACKS and TRANSFER ATTACKS ACROSS GOALS in `Experiments`.
```
cd eval
# bash eval.sh $model $attack $method
bash eval.sh llama2_chat_7b attngcg direct
```

- To evaluate the results of TRANSFER ATTACKS ACROSS MODELS, use `eval/keyword_detection/all_kw_classify.sh` and `eval/gpt4_judge/all_gpt_classify.sh` for keyword detection method or GPT-4 evaluation. `target_model` means the closed-source victim model.
```
# use keyword detection method to evaluate attack success rate
cd eval/keyword_detection\
# bash all_kw_classify.sh $DIR_TO_RESULTS
# bash all_kw_classify.sh ../../attack_closed_model/attack_${target_model}/generation
bash all_kw_classify.sh ../../attack_closed_model/attack_gemini_pro/generation

# use GPT-4 to evaluate attack success rate
cd eval/gpt4_judge
# bash all_gpt_classify.sh $DIR_TO_RESULTS
# bash all_gpt_classify.sh ../../attack_closed_model/attack_${target_model}/generation
bash all_gpt_classify.sh ../../attack_closed_model/attack_gemini_pro/generation
```

# Reproducibility
In order to reproduce the experiment, we provide bash scripts experiments/bash_scripts that can be used out of the box, and all running settings and hyperparameters are included in bash scripts and experiments/configs/.

A note for hardware: all experiments we run use one or multiple NVIDIA A100 GPUs, which have 80G memory per chip.

## License
`AttnGCG` is licensed under the terms of the MIT license. See LICENSE for more details.

## Citation
If you find our work useful to your research and applications, please consider citing the paper and staring the repo :)

```bibtex
@article{wang2024attngcgenhancingjailbreakingattacks,
      title={AttnGCG: Enhancing Jailbreaking Attacks on LLMs with Attention Manipulation}, 
      author={Zijun Wang and Haoqin Tu and Jieru Mei and Bingchen Zhao and Yisen Wang and Cihang Xie},
      year={2024},
      journal={arXiv preprint arXiv:2410.09040}
}
```

## Acknowledgement
This work is partially supported by a gift from Open Philanthropy. We thank the Center for AI Safety, NAIRR Pilot Program, the Microsoft Accelerate Foundation Models Research Program, and the OpenAI Researcher Access Program for supporting our computing needs. Any opinions, findings, conclusions, or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the sponsors' views.
