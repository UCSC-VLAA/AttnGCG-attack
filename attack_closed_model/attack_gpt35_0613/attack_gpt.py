import openai
import json
import argparse

openai.api_key = None

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Test the asr of generations")
    
    parser.add_argument(
        "--source_testcase_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--output_generation_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--icl_path",
        type=str,
        default=None
    )

    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    source_testcase_path = args.source_testcase_path
    output_generation_path = args.output_generation_path
    icl_path = args.icl_path

    print(source_testcase_path, flush=True)
    
    with open(source_testcase_path, 'r') as f:
        source_testcases = json.load(f)
    output_generations = {}
    output_icl = {}
    
    for goal in source_testcases.keys():
        print(goal, flush=True)
        output_generations[goal] = []
        output_icl[goal] = []
        testcases = source_testcases[goal]
        for testcase in testcases:
            inp = testcase
            print(inp)
            icl = {}
            icl['goal'] = goal
            icl['testcase'] = testcase
            result = completion_with_backoff(
                        engine="gpt-35-turbo-0613-nofilter",
                        messages=[
                        { "role": "system", "content": "You are a helpful assistant." },
                        { "role": "user", "content": inp}],
                        temperature=1e-8,
                        max_tokens=256)
            generation = result['choices'][0]['message']['content']
            icl['generation'] = generation
            output_generations[goal].append(generation)
            output_icl[goal].append(icl)
            with open(output_generation_path, 'w') as f:
                json.dump(output_generations, f, indent=4)
            with open(icl_path, 'w') as f:
                json.dump(output_icl, f, indent=4)
        with open(output_generation_path, 'w') as f:
            json.dump(output_generations, f, indent=4)
        with open(icl_path, 'w') as f:
            json.dump(output_icl, f, indent=4)
    
if __name__ == "__main__":
    main()
