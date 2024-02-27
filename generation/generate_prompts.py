import argparse
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import re

def clean_document(doc):
    cleaned_doc = re.sub(r'[^\x00-\x7F]+', '', doc)
    return cleaned_doc

def load_related_docs(related_doc_path):
    with open(related_doc_path, 'r') as ref_file:
        top_k_data = json.load(ref_file)
    return {entry['id']: [entry['sections'], entry['top-chunks']] for entry in top_k_data}

def construct_wikipedia_prompt(metadata, section, related_docs, tokenizer, max_input_length):
    keyword = metadata['key']
    prompt_1 = f"I have a topic \"{keyword}\" and a section \"{section}\" that contains the following documents:\n"
    prompt_2 = '\n'.join([f"Document {i+1}: {doc}" for i, doc in enumerate(related_docs)])
    prompt_3 = """\
\nBased on the above information, you are assigned to write the particular section of a Wikipedia article on the topic.
You must cite the most relevant document for every sentence you write, in the format of "This is an example sentence.[k]", where k denotes Document k. 
"""    
    tokens_1 = tokenizer.encode(prompt_1, add_special_tokens=False)
    tokens_2 = tokenizer.encode(prompt_2, add_special_tokens=False)
    tokens_3 = tokenizer.encode(prompt_3, add_special_tokens=False)
    tokens = tokens_1 + tokens_2[:max_input_length - len(tokens_1) - len(tokens_3)] + tokens_3
    prompt = tokenizer.decode(tokens)
    return "", prompt

def construct_outline_prompt(metadata, related_docs, tokenizer, max_input_length):
    keyword = metadata['key']

    prompt_1 = f"I have a topic \"{keyword}\" that contains the following documents:\n"

    prompt_2 = '\n'.join([f"Document {i+1}: {doc}" for i, doc in enumerate(related_docs)])
    
    prompt_3 = """\
\nBased on the above information, you are assigned to write an outline for a Wikipedia article about this topic.
Your outline should only include the names of the sections, without any further details.
Do not use document name as your outline.
The format of your outline should be as follows:
1. Introduction
2. <Section Name 1>
...
n. <Section Name n> 
"""    
    tokens_1 = tokenizer.encode(prompt_1, add_special_tokens=False)
    tokens_2 = tokenizer.encode(prompt_2, add_special_tokens=False)
    tokens_3 = tokenizer.encode(prompt_3, add_special_tokens=False)
    
    tokens = tokens_1 + tokens_2[:max_input_length - len(tokens_1) - len(tokens_3)] + tokens_3
    prompt = tokenizer.decode(tokens)

    return "", prompt

def save_prompt(metadata, sys_prompt_list, usr_prompt_list, prompts_dir, related_docs_list, sections_list):
    key = metadata['key']
    outline = sections_list
    prompt_data = {
        'key': key,
        'outline': outline,
        'retrieved_chunks': related_docs_list,
        'sys_prompt': sys_prompt_list,
        'usr_prompt': usr_prompt_list
    }
    prompt_file_path = os.path.join(prompts_dir, f"{metadata['id']}.json")
    with open(prompt_file_path, 'w') as out_file:
        json.dump(prompt_data, out_file, indent=4)

def generate_wikipedia_prompt(metadata_dir, related_doc_path, prompts_dir, model_path, max_input_length):
    if not os.path.exists(prompts_dir):
        os.makedirs(prompts_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    related_docs = load_related_docs(related_doc_path)
    exclude_section_titles = ["references", "citations", "see_also", "external_links", "notes", 'bibliography', 'further_reading']
    exclude_section_titles.extend(['see also', 'external links', 'further reading'])
    
    for filename in tqdm(os.listdir(metadata_dir)):
        if filename.endswith('.json'):
            metadata_path = os.path.join(metadata_dir, filename)
            with open(metadata_path, 'r') as file:
                metadata = json.load(file)
            
            metadata['outlines'] = [o for o in metadata['outlines'] if o.lower() not in exclude_section_titles]
            try:
                prefilter_sections_list = related_docs[metadata['id']][0]
                prefilter_related_docs_list = related_docs[metadata['id']][1]
                sections_list, related_docs_list = [], []
                for related_docs, section in zip(prefilter_related_docs_list, prefilter_sections_list):
                    if section.lower() in exclude_section_titles:
                        continue
                    sections_list.append(section)
                    related_docs_list.append(related_docs[:5])
            except KeyError:
                continue  # not in subset
            sys_prompt_list, usr_prompt_list = [], []
            for idx, related_docs in enumerate(related_docs_list):
                related_docs = [clean_document(doc) for doc in related_docs]
                sys_prompt, usr_prompt = construct_wikipedia_prompt(metadata, sections_list[idx], related_docs, tokenizer, max_input_length)
                sys_prompt_list.append(sys_prompt)
                usr_prompt_list.append(usr_prompt)
            save_prompt(metadata, sys_prompt_list, usr_prompt_list, prompts_dir, related_docs_list, sections_list)

def main():
    parser = argparse.ArgumentParser(description="Generate Wikipedia prompts from related documents.")
    parser.add_argument("--metadata_dir", type=str, default='data/metadata', help="Directory containing metadata JSON files.")
    parser.add_argument("--related_doc_path", type=str, default='data/retrieve_result/top-50-dpr-llama-7b-per-section-text.json', help="Path to the JSON file with related documents.")
    parser.add_argument("--prompts_dir", type=str, default='data/prompts/dpr_llama_7b_per_section', help="Directory to save the generated prompts.")
    parser.add_argument("--model_path", type=str, default='/home/junhao/models/meta-llama/Llama-2-7b-chat-hf', help="Path to the tokenizer model.")
    parser.add_argument("--max_input_length", type=int, default=2048, help="Maximum input length for the tokenizer.")
    args = parser.parse_args()

    generate_wikipedia_prompt(args.metadata_dir, args.related_doc_path, args.prompts_dir, args.model_path, args.max_input_length)

if __name__ == "__main__":
    main()
