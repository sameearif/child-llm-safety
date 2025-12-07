import json
import random

MODEL_MAP = {
    'google/gemma-3-12b-it': 'Gemma-3-12B',
    'google/gemma-3-27b-it': 'Gemma-3-27B',
    'meta-llama/Meta-Llama-3.1-8B-Instruct': 'Llama-3-8B',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo': 'Llama-3-70B',
    'openai/gpt-oss-20b': 'GPT-OSS-20B',
    'openai/gpt-oss-120b': 'GPT-OSS-120B',
    'deepseek-ai/DeepSeek-V3.1': 'DeepSeek-V3',
    'deepseek-ai/DeepSeek-R1-0528': 'DeepSeek-R1',
    'Qwen/Qwen3-14B': 'Qwen-3-14B',
    'Qwen/Qwen3-32B': 'Qwen-3-32B',
    'gpt-5.1': 'GPT-5.1',
    'claude-sonnet-4-5': 'Claude-Sonnet-4.5',
    'gemini-3-pro-preview': 'Gemini-3-Pro'
}

def load_dataset(file_name):
    dataset = json.load(open(f'datasets/{file_name}'))
    return dataset

def load_responses(model_name):
    responses = json.load(open(f'responses/{MODEL_MAP[model_name]}.json'))
    return responses

def append_age(dataset):
    new_dataset = {}
    for category in dataset:
        new_dataset[category] = []
        for prompt in dataset[category]:
            age = random.randint(6, 13)
            new_dataset[category].append(f'Age: {age}\n\n{prompt}')
    return new_dataset
