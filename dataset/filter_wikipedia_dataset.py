import argparse
from datasets import load_dataset

def filter_dataset(old_version, new_version, min_length, max_length, output_dir):
    """
    Filters Wikipedia datasets based on titles and text length.

    Args:
    - old_version (str): Version of the older Wikipedia dataset.
    - new_version (str): Version of the newer Wikipedia dataset.
    - min_length (int): Minimum length of the text.
    - max_length (int): Maximum length of the text.
    - output_dir (str): Directory to save the filtered datasets.
    """
    old_dataset = load_dataset("wikipedia", old_version)
    new_dataset = load_dataset("wikimedia/wikipedia", new_version)
    
    old_titles = set(old_dataset["train"]["title"])
    new_titles = set(new_dataset["train"]["title"])
    
    unique_titles_2023 = new_titles.difference(old_titles)
    
    dataset_2023 = new_dataset.filter(lambda x: x["title"] in unique_titles_2023)
    dataset_2023.save_to_disk(f"{output_dir}/wikipedia2023set")
    
    filtered_dataset = dataset_2023.filter(lambda x: min_length <= len(x["text"].split()) <= max_length)
    filtered_dataset.save_to_disk(f"{output_dir}/wiki2023len1to3k")

def main():
    parser = argparse.ArgumentParser(description="Filter Wikipedia datasets based on titles and text length.")
    parser.add_argument("--old_version", type=str, default="20220301.en", help="Version of the older Wikipedia dataset.")
    parser.add_argument("--new_version", type=str, default="20231101.en", help="Version of the newer Wikipedia dataset.")
    parser.add_argument("--min_length", type=int, default=1000, help="Minimum length of the text.")
    parser.add_argument("--max_length", type=int, default=3000, help="Maximum length of the text.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the filtered datasets.")
    
    args = parser.parse_args()

    filter_dataset(args.old_version, args.new_version, args.min_length, args.max_length, args.output_dir)

if __name__ == "__main__":
    main()
