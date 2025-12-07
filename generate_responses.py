import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from utils import load_dataset, append_age, MODEL_MAP

def generate_responses(client, model, dataset):
    try:
        with open(f'responses/{MODEL_MAP[model]}.json', 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except:
        responses = {}
    for category in dataset:
        if category not in responses:
            responses[category] = []
        for i, prompt in tqdm(enumerate(dataset[category]), total=len(dataset[category])):
            if i < len(responses[category]):
                continue
            response = client.chat.completions.create(
            model=model,
            messages=[
                    {'role': 'user', 'content': prompt}
                ]
            )
            response = response.choices[0].message.content
            responses[category].append({'prompt': prompt, 'response': response.split('</think>')[-1]})
            with open(f'responses/{MODEL_MAP[model]}.json', 'w', encoding='utf-8') as f:
                json.dump(responses, f, indent='\t', ensure_ascii=False)

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
                'Qwen/Qwen3-32B'
            ],
        required=True, help='Model name, e.g., gpt-5 or llama-3')
    parser.add_argument('--dataset_name', type=str, choices=["CSB.json", "CSB_Cues.json"], required=True, help='Dataset file name')
    parser.add_argument('--add_age', type=bool, help='Explicitly add age in the prompt')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)
    if args.add_age:
        dataset = append_age(dataset)

    client = OpenAI(
        api_key=os.getenv('DEEPINFRA_API_KEY'),
        base_url='https://api.deepinfra.com/v1/openai',
    )
    generate_responses(client, args.model, dataset)

if __name__ == "__main__":
    main()