import requests
import json
import os
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def send_request(sys_prompt, usr_prompt, model_name, completion_url):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt}
    ] if sys_prompt else [{"role": "user", "content": usr_prompt}]
    
    response = requests.post(
        completion_url,
        json={"model": model_name, "messages": messages, "max_tokens": 4096}
    )
    return response.json()

def parse_result(response):
    if 'choices' in response and len(response['choices']) > 0:
        return response['choices'][0]['message']['content'].strip()
    else:
        return ''

def process_single_file(prompt_path, output_path, model_name, completion_url):
    with open(prompt_path, 'r') as file:
        prompt_data = json.load(file)

    sys_prompts = prompt_data.get('sys_prompt', [])
    usr_prompts = prompt_data.get('usr_prompt', [])
    
    generated_contents = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_response = {
            executor.submit(send_request, sys_prompt, usr_prompt, model_name, completion_url): index
            for index, (sys_prompt, usr_prompt) in enumerate(zip(sys_prompts, usr_prompts))
        }
        
        for future in as_completed(future_to_response):
            index = future_to_response[future]
            try:
                response = future.result()
                generated_content = parse_result(response)
                generated_contents.append((index, generated_content))
            except Exception as exc:
                print(f'Request failed with {exc}')
                generated_contents.append((index, ''))

    # Sort generated contents back into their original order
    generated_contents.sort(key=lambda x: x[0])
    prompt_data['generated_content'] = [content for _, content in generated_contents]

    # Prepare data for saving
    jsonl_data = [
        {
            "key": prompt_data["key"],
            "section": section,
            "retrieved_chunks": chunks,
            "sys_prompt": sys_prompt,
            "usr_prompt": usr_prompt,
            "generated_content": gen
        }
        for gen, section, chunks, sys_prompt, usr_prompt in zip(
            prompt_data["generated_content"], prompt_data["outline"], 
            prompt_data["retrieved_chunks"], prompt_data["sys_prompt"], 
            prompt_data["usr_prompt"]
        )
    ]

    # Save each item in jsonl format
    with open(output_path + 'l', 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')

def process_and_save_results(prompts_dir, model_name, completion_url):
    output_dir = f'data/generated/{model_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm([f for f in os.listdir(prompts_dir) if f.endswith('.json')]):
        prompt_path = os.path.join(prompts_dir, filename)
        output_path = os.path.join(output_dir, filename.split('.')[0])  # Adjust filename for .jsonl output
        if os.path.exists(output_path + '.jsonl'):  # Adjusted to check for .jsonl file
            print(f"Skipping {filename} as it already exists.")
            continue

        process_single_file(prompt_path, output_path, model_name, completion_url)

def main():
    parser = argparse.ArgumentParser(description='Process prompts and save generated responses.')
    parser.add_argument('--prompts_dir', type=str, default='data/prompts', help='Directory containing prompt JSON files.')
    parser.add_argument('--model_name', type=str, default='vicuna-7b-v1.5', help='Name of the model to use for generation.')
    parser.add_argument('--completion_url', type=str, default='http://localhost:8010/v1/chat/completions', help='URL for the model completion API endpoint.')
    args = parser.parse_args()
    
    process_and_save_results(args.prompts_dir, args.model_name, args.completion_url)

if __name__ == "__main__":
    main()
