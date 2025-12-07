import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
from utils import load_dataset, MODEL_MAP

def get_review(reviewer, prompt, response, system_prompt):
    response = reviewer.chat.completions.create(
        model='gpt-5.1',
        messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'Question:\n{prompt}\n\n\nResponse:\n{response}'}
            ]
    )
    response = json.loads(response.choices[0].message.content)
    return response

def generate_responses(generator, reviewer, model, dataset, system_prompt_generator, system_prompt_reviewer):
    try:
        with open(f'gold-responses/{MODEL_MAP[model]}.json', 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except:
        responses = {}
    for category in dataset:
        if category not in responses:
            responses[category] = []
        for i, prompt in tqdm(enumerate(dataset[category]), total=len(dataset[category])):
            if i < len(responses[category]):
                continue
            messages = [
                {'role': 'system', 'content': system_prompt_generator},
                {'role': 'user', 'content': prompt}
            ]
            response = generator.chat.completions.create(
                model=model,
                messages=messages,
            )
            response = response.choices[0].message.content.split('</think>')[-1]
            messages.append({'role': 'assistant', 'content': response})
            review = get_review(reviewer, prompt, response, system_prompt_reviewer)
            updated = False
            if sum(int(v) for k, v in review.items() if k != "reasoning") < 10:
                messages.append({'role': 'user', 'content': f'Response Review:\n{str(review)}\n\nUpdate the response based on the review. Only return the updated response without any extra commentary or reference to the review/score.'})
                response = generator.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                response = response.choices[0].message.content.split('</think>')[-1]
                updated = True
            responses[category].append({'prompt': prompt, 'initial_response': messages[2]['content'], 'final_response': response, 'review': review, 'updated': updated})
            with open(f'gold-responses/{MODEL_MAP[model]}.json', 'w', encoding='utf-8') as f:
                json.dump(responses, f, indent='\t', ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['gpt-5.1', 'claude-sonnet-4-5', 'gemini-3-pro-preview'], required=True, help='Model name, e.g., gpt-5 or llama-3')
    parser.add_argument('--dataset_name', type=str, choices=["CSB.json", "CSB_Cues.json"], required=True, help='Dataset file name')
    args = parser.parse_args()

    env = Environment(loader=FileSystemLoader('prompts'))
    system_prompt_generator = env.get_template('gold-response.jinja').render()
    system_prompt_reviewer = env.get_template('llm-judge.jinja').render()
    dataset = load_dataset(args.dataset_name)
    reviewer = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    if args.model == 'gpt-5.1':
        generator = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    elif args.model == 'claude-sonnet-4-5':
        generator = OpenAI(api_key=os.environ.get('ANTHROPIC_API_KEY'), base_url="https://api.anthropic.com/v1/")
    elif args.model == 'gemini-3-pro-preview':
        generator = OpenAI(api_key=os.environ.get('GEMINI_API_KEY'), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    generate_responses(generator, reviewer, args.model, dataset, system_prompt_generator, system_prompt_reviewer)

if __name__ == "__main__":
    main()
