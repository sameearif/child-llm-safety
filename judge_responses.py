import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
from utils import load_responses, MODEL_MAP

def judge_responses(client, responses, system_prompt, file_name):
    try:
        with open(f'evaluation/{MODEL_MAP[file_name]}.json', 'r', encoding='utf-8') as f:
            evaluaton = json.load(f)
    except:
        evaluaton = {}
    for category in responses:
        if category not in evaluaton:
            evaluaton[category] = []
        for i, item in tqdm(enumerate(responses[category]), total=len(responses[category])):
            if i < len(evaluaton[category]):
                continue
            response = client.chat.completions.create(
                model='gpt-5.1',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f'Question:\n{item["prompt"]}\n\n\nResponse:\n{item["response"]}'}
                ]
            )
            evaluaton[category].append({'prompt': item['prompt'], 'response': item['response'], 'evaluation': json.loads(response.choices[0].message.content)})
            with open(f'evaluation/{MODEL_MAP[file_name]}.json', 'w', encoding='utf-8') as f:
                json.dump(evaluaton, f, indent='\t', ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=[
                'google/gemma-3-12b-it',
                'google/gemma-3-27b-it',
                'meta-llama/Meta-Llama-3.1-8B-Instruct',
                'meta-llama/Llama-3.3-70B-Instruct-Turbo',
                'openai/gpt-oss-20b',
                'openai/gpt-oss-120b',
                'deepseek-ai/DeepSeek-V3.1',
                'deepseek-ai/DeepSeek-R1-0528',
                'Qwen/Qwen3-14B',
                'Qwen/Qwen3-32B',
                'sameearif/LlamaPlushie-3.1-8B-KTO',
                'sameearif/LlamaPlushie-3.1-8B-DPO',
                'sameearif/LlamaPlushie-3.1-8B-SFT'
            ],
        required=True, help='Model name, e.g., gpt-5 or llama-3')
    args = parser.parse_args()
    
    responses = load_responses(args.model)
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    env = Environment(loader=FileSystemLoader('prompts'))
    system_prompt = env.get_template('llm-judge.jinja').render()

    judge_responses(client, responses, system_prompt, args.model)
    
if __name__ == "__main__":
    main()
