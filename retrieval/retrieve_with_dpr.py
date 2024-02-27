import argparse
import json
import glob
import torch
import os
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from tqdm import tqdm
import gc

def load_passages(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return [(entry['chunk_id'], entry['content']) for entry in data]

def load_queries(folder_path):
    queries = {}
    for file_path in glob.glob(f"{folder_path}/*.json"):
        with open(file_path, 'r') as file:
            data = json.load(file)
            queries[data['id']] = data['key']
    return queries

def load_sections(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def main(docs_dir, metadata_dir, sections_file, embeddings_dir, output_file, model_path, cuda_device, k):
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")

    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_path)
    question_model = DPRQuestionEncoder.from_pretrained(model_path)
    question_model = question_model.to(device).eval()

    queries = load_queries(metadata_dir)
    sections_dict = load_sections(sections_file)
    results = []
    total, cnt = 0, 0

    for query_id, query_title in tqdm(queries.items(), desc="calculating similarities"):
        context_embeddings_path = os.path.join(embeddings_dir, f"{query_id}.pt")
        if not os.path.exists(context_embeddings_path):
            continue  # Skip if embeddings are not available

        context_embeddings = torch.load(context_embeddings_path).to(device)
        passages = load_passages(os.path.join(docs_dir, f"{query_id}.json"))

        try:
            sections = sections_dict[str(query_id)]  # subset
            cnt += 1
        except KeyError:
            continue  # Skip if no sections are available for this query

        process_query(query_id, query_title, sections, context_embeddings, passages, question_model, question_tokenizer, device, results, k)

    with open(output_file, 'w') as outfile:
        json.dump(results, outfile)
    
    print(total / max(cnt, 1))
    gc.collect()
    torch.cuda.empty_cache()

def process_query(query_id, query_title, sections, context_embeddings, passages, question_model, question_tokenizer, device, results, k):
    top_chunk_ids_per_section = []
    sec_set = set()

    for section in sections:
        with torch.no_grad():
            # Encode the query for the section
            encoded_query = question_tokenizer(f'{section.lower()} of {query_title.lower()}?', return_tensors='pt', truncation=True, max_length=512).to(device)
            query_embedding = question_model(**encoded_query).pooler_output

            # Compute similarities between the query embedding and all context embeddings
            similarities = (query_embedding @ context_embeddings.T).squeeze(0)

            # Retrieve top-k passages based on the similarities
            top_k = min(k, context_embeddings.size(0))
            top_results = similarities.topk(top_k)

            # Extract the top chunk IDs
            prefilter_top_chunk_ids = [passages[idx][0] for idx in top_results.indices.cpu().numpy()]
            top_chunk_ids = prefilter_top_chunk_ids[:5]  # Adjust as necessary for the number of top passages to consider

            for id in top_chunk_ids:
                sec_set.add(id)
            top_chunk_ids_per_section.append(top_chunk_ids)

    # Aggregate the results for this query
    results.append({'id': query_id, 'sections': sections, 'top-chunks': top_chunk_ids_per_section})

    # Optionally, update the total and count for average calculation (if needed outside the function)
    return len(sec_set)  # Return the count of unique sections for this query


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings and retrieve top passages for each query section.")
    parser.add_argument("--docs_dir", type=str, default="data/doc/chunked", help="Directory containing document JSON files.")
    parser.add_argument("--metadata_dir", type=str, default="data/metadata", help="Directory containing metadata JSON files.")
    parser.add_argument("--outline_file", type=str, default="vicuna-7b_outline.json", help="JSON file containing section outlines.")
    parser.add_argument("--embeddings_dir", type=str, default="dpr_context_embeddings", help="Directory to load stored context embeddings from.")
    parser.add_argument("--output_file", type=str, default="top-50-dpr-vicuna-7b.json", help="File to save the retrieval results.")
    parser.add_argument("--model_path", type=str, default="facebook/dpr-question_encoder-multiset-base", help="Directory containing the DPR model.")
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device to use for model inference.")
    parser.add_argument("--docs_num", type=int, default=50, help="Number of top related documents retrieved.")
    args = parser.parse_args()

    main(args.docs_dir, args.chunks_dir, args.metadata_dir, args.outline_file, args.embeddings_dir, args.output_file, args.model_path, args.cuda_device, args.docs_num)
