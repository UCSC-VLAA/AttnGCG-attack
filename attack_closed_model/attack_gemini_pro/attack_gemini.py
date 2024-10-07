import textwrap

import google.generativeai as genai

from IPython.display import Markdown
from google.generativeai.types import HarmCategory, HarmBlockThreshold


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
GOOGLE_API_KEY = None

import json
import argparse

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))
def completion_with_backoff(model, prompt):
    response = model.generate_content(prompt)
    return response

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
    
    generation_config = {
        "temperature": 1e-8,
        "max_output_tokens": 2048
    }
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    }
    error_count = 0
    keys_num = len(GOOGLE_API_KEY)
    i = 0
    
    for goal in source_testcases.keys():
        print(goal, flush=True)
        output_generations[goal] = []
        output_icl[goal] = []
        testcases = source_testcases[goal]
        for testcase in testcases:
            if i == keys_num:
                i = 0
            genai.configure(api_key=GOOGLE_API_KEY[i])
            i += 1
            model = genai.GenerativeModel(model_name='gemini-pro',
                                            generation_config=generation_config,
                                            safety_settings=safety_settings)
            inp = testcase
            print(inp)
            icl = {}
            icl['goal'] = goal
            icl['testcase'] = testcase
            response = completion_with_backoff(model=model,
                        prompt= inp)
            try:
                generation = response.text
            except:
                if response.candidates == []:
                    print(f"response.prompt_feedback:\n{response.prompt_feedback}")
                else:
                    print(f"response.prompt_feedback:\n{response.prompt_feedback}")
                    print(f"response.candidates:\n{response.candidates}")
                error_count += 1
                continue
            
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
        print(f"error_count: {error_count}")
    
if __name__ == "__main__":
    main()
