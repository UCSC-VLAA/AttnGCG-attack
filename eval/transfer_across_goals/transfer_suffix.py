import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Sample the train data")
    parser.add_argument(
        "--model",
        type=str,
        default='llama2_chat_7b',
        help="target model"
    )
    parser.add_argument(
        "--attack",
        type=str,
        default='attngcg',
        choices=['attngcg', 'gcg'],
        help="attack type"
    )
    parser.add_argument(
        "--method",
        type=str,
        default='transfer',
        help="experiment type"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = args.model
    attack = args.attack
    method = args.method
    train_goals_path = f"blank_transfer_train_25.json"
    test_goals_path = f"blank_transfer_test_100.json"
    suffix_path = f"raw_testcase/{model}_{attack}_{method}.json"

    with open(suffix_path ,'r') as f:
        content = json.load(f)
    first_goal = list(content.keys())[0]
    prompts = content[first_goal]
    suffixes = [prompt[len(first_goal):] for prompt in prompts]
        
    train_prompt_path = f"use_testcase/{model}_{attack}_{method}_train_25.json"
    test_prompt_path = f"use_testcase/{model}_{attack}_{method}_test_100.json"

    file_pairs = [(train_goals_path, train_prompt_path), (test_goals_path, test_prompt_path)]

    for source_file, dest_file in file_pairs: 
        with open(source_file ,'r') as f:
            source = json.load(f)
        dest = {}
        for goal in source.keys():
            dest[goal] = [goal + suffix for suffix in suffixes]
        with open(dest_file, 'w') as f:
            json.dump(dest, f, indent=4)
        

if __name__ == "__main__":
    main()