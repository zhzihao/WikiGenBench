import os
import json
import argparse
from mediawiki import MediaWiki
from datasets import load_from_disk
from tqdm import tqdm
import concurrent.futures

def read_finished_ids(output_dir):
    """Reads IDs of already processed pages to avoid reprocessing."""
    finished_ids = set()
    for filename in os.listdir(output_dir):
        try:
            finished_id = int(filename.split('.')[0])
            finished_ids.add(finished_id)
        except ValueError:
            continue  # Skip files that do not have an integer ID as their name
    return finished_ids

def save_page_data(page_data, output_dir):
    """Saves the extracted page data to a JSON file."""
    pageid = page_data['id']
    with open(os.path.join(output_dir, f'{pageid}.json'), 'w', encoding='utf-8') as f:
        json.dump(page_data, f, ensure_ascii=False, indent=4)

def process_page(data, wikipedia, output_dir, finished_ids):
    """Processes a single Wikipedia page by its ID."""
    pageid = int(data['id'])
    if pageid in finished_ids:
        return  # Skip already processed pages

    try:
        p = wikipedia.page(pageid=pageid)
        page_data = {
            'id': pageid,
            'title': p.title,
            'summary': p.summary,
            'sections': p.sections,
            'content': p.content,
            'content_by_section': [p.section(sec) for sec in p.sections],
            'references': p.references,
            'wikitext': p.wikitext
        }
        save_page_data(page_data, output_dir)
    except Exception as e:
        print(f"Error processing page id {pageid}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process Wikipedia pages and save metadata.")
    parser.add_argument("--dataset_path", type=str, default="data/Wikipedia2023-len-1k-to-3k/train", help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="data/metadata", help="Directory to save processed page data")
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)
    wikipedia = MediaWiki()
    os.makedirs(args.output_dir, exist_ok=True)
    finished_ids = read_finished_ids(args.output_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_page, data, wikipedia, args.output_dir, finished_ids) for data in dataset]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()  # Handling exceptions inside process_page

if __name__ == "__main__":
    main()
