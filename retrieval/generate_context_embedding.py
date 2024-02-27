import argparse
import json
import glob
import torch
import os
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from tqdm import tqdm
import gc

def main(metadata_dir, docs_dir, embeddings_dir, context_model_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_model_path)
    context_model = DPRContextEncoder.from_pretrained(context_model_path).to(device).eval()

    def load_passages(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def load_queries(folder_path):
        queries = {}
        for file_path in glob.glob(f"{folder_path}/*.json"):
            with open(file_path, 'r') as file:
                data = json.load(file)
                queries[data['id']] = data['key']
        return queries

    def encode_passages_in_batches(passages, batch_size):
        batched_embeddings = []
        for i in range(0, len(passages), batch_size):
            batch = [p['content'] for p in passages[i:i + batch_size]]
            with torch.no_grad():
                encoded_contexts = context_tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
                context_outputs = context_model(**encoded_contexts)
                batched_embeddings.append(context_outputs.pooler_output.cpu())
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return torch.cat(batched_embeddings, dim=0)

    queries = load_queries(metadata_dir)
    os.makedirs(embeddings_dir, exist_ok=True)

    for query_id, query_info in tqdm(queries.items(), desc="Encoding passages"):
        output_file = os.path.join(embeddings_dir, f"{query_id}.pt")
        if os.path.exists(output_file):
            print(f"Skipping {query_id}, already processed.")
            continue
        
        passages = load_passages(os.path.join(docs_dir, f"chunked/{query_id}.json"))
        context_embeddings = encode_passages_in_batches(passages, batch_size)
        torch.save(context_embeddings, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode context passages to embeddings.")
    parser.add_argument("--metadata_dir", type=str, default="data/metadata", help="Directory containing metadata JSON files.")
    parser.add_argument("--docs_dir", type=str, default="data/doc", help="Base directory containing document JSON files.")
    parser.add_argument("--embeddings_dir", type=str, default="dpr_context_embeddings", help="Directory to save generated embeddings.")
    parser.add_argument("--context_model_path", type=str, default="facebook/dpr-ctx_encoder-multiset-base", help="Model path for the DPR context encoder.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for encoding passages.")
    args = parser.parse_args()

    main(args.metadata_dir, args.docs_dir, args.embeddings_dir, args.context_model_path, args.batch_size)
