# WikiGenBench

## Dataset Generation

### Filtering Wikipedia Dataset

Start by filtering the Wikipedia dataset to match specific date and length criteria.

```bash
./dataset/filter_wikipedia_dataset.py --output_dir data
```

### Processing Wikipedia Dataset

Process the filtered dataset to prepare metadata.

```bash
./dataset/process_wikipedia_parser.py --data_path data/Wikipedia2023-len-1k-to-3k/train --output_dir data/metadata
```

### Collecting Related Documents Links

To enrich your dataset with high-quality references, collect links either from the reference section of Wikipedia articles or by using Google's API. The collected links should be formatted as follows:

```json
{
  "title": "The Title of the Article",
  "url": "https://example.com/link-to-the-article",
  "source": "Wikipedia/Google"
}
```

This step is crucial for gathering comprehensive background information and supporting materials for the dataset. Store these links in a structured format, as they will be used in subsequent steps for scraping and analysis.

### Scraping Links

Scrape the collected links to gather the data.

```bash
./dataset/scrape_links.py --input_dir data/search_link --output_dir data/scraped_data
```

### Data Cleaning and Chunking

After initial filtering and data cleaning, it's essential to organize the dataset for further processing and analysis. The data should be managed in a structured format as follows:

```json
{
  "doc_id": "Unique Document Identifier",
  "content": "The full text content of the document"
}
```

To facilitate more efficient processing and retrieval, large documents should be chunked into segments that can be processed individually by the system:

```bash
./dataset/chunk_docs.py --input_dir data/doc --output_dir data/doc/chunked
```

## Outline and Content Generation

Our framework leverages FastChat in conjunction with open-source large language models (LLMs) for generating text. Additionally, for testing and comparison purposes, we utilize OpenAI's API to generate text using GPT-3.5.

### Generating Prompts

The first step in the text generation process is to create prompts that will guide the model in producing the desired content. These prompts are crafted to encapsulate the context and specify the information or narrative style we aim to generate.

To generate prompts, use the following script:

```bash
./generation/generate_prompts.py
```

### Generating Responses with FastChat

FastChat is employed to generate text responses based on prompts derived from the dataset. Before generating responses, it's crucial to prepare outlines for the RRPR (Rapid Response Preparation Routine) process. These outlines help structure the generation process and ensure that the responses are organized and relevant.

### Generating Outlines for RRPR

Outlines are generated in the following format, capturing the structure of the content to be generated:

```json
{
  "pageid1": ["section_name1", "section_name2", ...],
  ...
}
```

## Retrieval

For efficient and accurate retrieval of relevant documents, our framework adopts the Dense Passage Retrieval (DPR) methodology.

### Generating Context Embeddings

The initial step in the DPR process involves generating embeddings for the documents. These embeddings represent documents in a high-dimensional vector space, enabling the calculation of relevance scores between documents and queries.

To generate context embeddings, run the following command:

```bash
./retrieval/generate_context_embedding.py --metadata_dir data/metadata --docs_dir data/doc/chunked --embeddings_dir dpr_context_embeddings
```

### Retrieving with DPR

With the context embeddings generated, the next step is to retrieve the top-k documents related to a given query or set of queries. This is achieved by calculating similarity scores between the query embeddings and document embeddings, typically using the dot product as a measure of similarity.

To retrieve documents using DPR, execute the following command:

```bash
./retrieve_with_dpr.py --metadata_dir data/metadata --docs_dir data/doc/chunked --embeddings_dir dpr_context_embeddings --outline_file vicuna-7b_outline.json --docs_num 50 --output_file top-50-dpr-vicuna-7b.json
```

The result provides a ranked list of documents based on their relevance to the query.
