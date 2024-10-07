import json
import pandas as pd
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Sample the train data")
    parser.add_argument(
        "--method",
        type=str,
        default='direct',
        choices=['direct', 'ica', 'autodan', 'transfer'],
        help="attack type"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    method = args.method
    if method in ['ica', 'autodan']:
        number = 100
        source_path = f"advbench/harmful_behaviors_{method}.csv"
        data = pd.read_csv(source_path)  
        dest_path = f"advbench/blank_{method}_{number}.json"
        goals = data['goal'].tolist()[:number]
        dest = {}
        for goal in goals:
            dest[goal] = []
        with open(dest_path, 'w') as f:
            json.dump(dest, f, indent=4)
    elif method == "direct":
        number = 100
        source_path = f"advbench/harmful_behaviors.csv"
        data = pd.read_csv(source_path)  
        dest_csv_path = f"advbench/harmful_behaviors_{method}.csv"
        dest_path = f"advbench/blank_{method}_{number}.json"
        indices = random.sample(range(len(data['goal'])), number)
        dest_data = data.iloc[indices]
        dest_data.to_csv(dest_csv_path, index=False,sep=',')
        all_goals = data['goal'].tolist()
        goals = [all_goals[i] for i in indices]
        dest = {}
        for goal in goals:
            dest[goal] = []
        with open(dest_path, 'w') as f:
            json.dump(dest, f, indent=4)
    elif method == "transfer":
        test_number = 100
        source_path = f"advbench/harmful_behaviors.csv"
        data = pd.read_csv(source_path) 
        dest_csv_path = f"advbench/harmful_behaviors_{method}.csv"
        train_dest_path = f"advbench/blank_transfer_train_25.json"
        test_dest_path = f"advbench/blank_transfer_test_{test_number}.json"
        train_goals = data['goal'].tolist()[:25]
        train_dest = {}
        for goal in train_goals:
            train_dest[goal] = []
        with open(train_dest_path, 'w') as f:
            json.dump(train_dest, f, indent=4)
        test_dest = {}
        test_indices = random.sample(range(25, len(data['goal'])), test_number)
        all_goals = data['goal'].tolist()
        test_goals = [all_goals[i] for i in test_indices]
        for goal in test_goals:
            test_dest[goal] = []
        with open(test_dest_path, 'w') as f:
            json.dump(test_dest, f, indent=4)    
        for i in range(25):
            test_indices.append(i)
        test_indices.sort()
        dest_data = data.iloc[test_indices]
        dest_data.to_csv(dest_csv_path, index=False,sep=',')
        

if __name__ == "__main__":
    main()