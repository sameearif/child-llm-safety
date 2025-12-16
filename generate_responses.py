import os
import json
import argparse
import torch
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download
from utils import load_dataset, append_age, MODEL_MAP

LORA_BASE_MAP = {
    'sameearif/LlamaPlushie-3.1-8B-KTO': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'sameearif/LlamaPlushie-3.1-8B-DPO': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'sameearif/LlamaPlushie-3.1-8B-SFT': 'meta-llama/Meta-Llama-3.1-8B-Instruct'
}

def generate_responses(client, model, dataset):
    try:
        filename = f'responses/{MODEL_MAP.get(model, model.split("/")[-1])}.json'
        with open(filename, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except:
        responses = {}
        
    for category in dataset:
        if category not in responses:
            responses[category] = []
        for i, prompt in tqdm(enumerate(dataset[category]), total=len(dataset[category]), desc=f"API: {category}"):
            if i < len(responses[category]):
                continue
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                response_text = response.choices[0].message.content
                responses[category].append({'prompt': prompt, 'response': response_text.split('</think>')[-1]})
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(responses, f, indent='\t', ensure_ascii=False)
            except Exception as e:
                print(f"Error: {e}")

def generate_responses_local(model_name, dataset):
    if model_name in LORA_BASE_MAP:
        base_model = LORA_BASE_MAP[model_name]
        enable_lora = True
        lora_path = snapshot_download(repo_id=model_name)
    else:
        base_model = model_name
        enable_lora = False
        lora_path = None

    llm = LLM(
        model=base_model,
        enable_lora=enable_lora,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        gpu_memory_utilization=0.90
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=1024,
        stop_token_ids=[
            tokenizer.eos_token_id, 
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    )

    filename = f'responses/{MODEL_MAP[model_name]}.json'
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except:
        responses = {}

    for category in dataset:
        if category not in responses:
            responses[category] = []
        
        prompts = dataset[category]
        start_idx = len(responses[category])
        new_prompts = prompts[start_idx:]
        
        if not new_prompts:
            continue

        formatted_prompts = []
        for p in new_prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(text)

        lora_req = LoRARequest("adapter", 1, lora_path) if enable_lora else None
        
        outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_req)

        for original_prompt, output in zip(new_prompts, outputs):
            generated_text = output.outputs[0].text
            clean_response = generated_text.split('</think>')[-1].strip()
            responses[category].append({'prompt': original_prompt, 'response': clean_response})

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent='\t', ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, choices=["CSB.json", "CSB_Cues.json", "CSB_FT_Eval.json"], required=True)
    parser.add_argument('--local', type=bool, default=False)
    parser.add_argument('--add_age', type=bool, default=False)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)
    if args.add_age:
        dataset = append_age(dataset)

    if not args.local:
        client = OpenAI(
            api_key=os.getenv('DEEPINFRA_API_KEY'),
            base_url='https://api.deepinfra.com/v1/openai',
        )
        generate_responses(client, args.model, dataset)
    else:
        generate_responses_local(args.model, dataset)

if __name__ == "__main__":
    main()