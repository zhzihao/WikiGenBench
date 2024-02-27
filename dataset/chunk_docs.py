import json
import os
import argparse
from nltk.tokenize import word_tokenize
from tqdm import tqdm

def chunk_text(text, chunk_size):
    """
    Splits text into chunks of a specified size.
    """
    words = word_tokenize(text)
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def process_json_files(input_dir, output_dir, chunk_size):
    """
    Processes JSON files in a directory, chunking the contents of each file and saving the results.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each JSON file in the input directory
    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)

            # Safely read the JSON file
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file_path}: {e}")
                continue
            except FileNotFoundError as e:
                print(f"File not found: {e}")
                continue

            chunked_data = []
            chunk_id = 0  # Initialize chunk ID

            # Process each document in the file
            for document in data:
                doc_id = document.get('doc_id')
                content = document.get('content', '')

                # Chunk the content
                chunks = chunk_text(content, chunk_size)

                # Append chunks to chunked_data with continuous chunk IDs
                for chunk in chunks:
                    chunked_data.append({'chunk_id': chunk_id, 'doc_id': doc_id, 'content': chunk})
                    chunk_id += 1

            # Write the chunked data to a new JSON file in the output directory
            output_file_name = f'{os.path.splitext(file_name)[0]}.json'
            output_file_path = os.path.join(output_dir, output_file_name)
            try:
                with open(output_file_path, 'w') as output_file:
                    json.dump(chunked_data, output_file, indent=4)
            except IOError as e:
                print(f"Error writing to file {output_file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Chunk text from JSON files.")
    parser.add_argument("--input_dir", type=str, default="data/doc", help="Directory containing the JSON files to process")
    parser.add_argument("--output_dir", type=str, default="data/doc/chunked", help="Directory to save chunked files")
    parser.add_argument("--chunk_size", type=int, default=256, help="Number of words per chunk")
    args = parser.parse_args()

    process_json_files(args.input_dir, args.output_dir, args.chunk_size)
    print("Chunking process completed.")

if __name__ == "__main__":
    main()
