import os
import json
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader

def generate_prompts(client, prompt):
    response = client.chat.completions.create(
        model='gpt-5.1',
        messages=[
            {'role': 'user', 'content': prompt},
        ]
    )
    prompts = json.loads(response.choices[0].message.content)
    return prompts

def main():
    categories = ['Medical/Health', 'Sexual', 'Harm/Bullying/Substance (Drugs)/Online', 'Classroom Advice (e.g skipping class)', 'Family/Friends', 'Moral/Values', 'Hateful/stereotypes']
    keys = ['health', 'sexual', 'harm', 'classroom', 'relationships', 'morals', 'hateful']

    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    
    env = Environment(loader=FileSystemLoader('prompts'))
    dataset = {}
    for i, category in enumerate(categories):
        prompt = env.get_template('data-generation.jinja').render(category=category)
        prompts = generate_prompts(client, prompt)
        dataset[keys[i]] = prompts
    with open('datasets/CSB.json', 'w') as f:
        json.dump(dataset, f, indent='\t', ensure_ascii=False)

if __name__ == "__main__":
    main()
