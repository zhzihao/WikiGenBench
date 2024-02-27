import os, re, json
import argparse
from tqdm import tqdm
from prompt import *
from utils import max_tokens, count_tokens, get_top_related_docs
from callgpt import chatgpts

class Writer(object):
    def __init__(self, prompt):
        self.prompt = prompt

    def __call__(self, key, subtitle=None, refs=None):
        user_prompt = f"Keyword: {key}\n"
        if subtitle:
            if isinstance(subtitle, str): user_prompt += f"Chapter: {subtitle}\n"
            elif isinstance(subtitle, list): user_prompt += f"Outline: {', '.join(subtitle)}"
        if refs: user_prompt += "\n".join(refs) + "\n"
        return [{"role": "system", "content": self.prompt}, {"role": "user", "content": user_prompt}]

def clean_outline(outlines):
    ref_chapters = set(["references", "see also", "notes", "external links", "bibliography", "further reading"])
    cleaned_outlines = []
    for subtitle in outlines:
        subtitle = subtitle.replace("_", " ")
        if subtitle.lower() not in ref_chapters: cleaned_outlines.append(subtitle)
    return cleaned_outlines

def vanilla(level, num, token, given_outline=False): # no reference provided
    writer = Writer(prompt_vanilla_given_outline) if given_outline else Writer(prompt_vanilla)
    filename, prompts = [], []
    for file in tqdm(os.listdir("data")):
        if os.path.exists(os.path.join("baseline", f"{level}_{num}", file)): continue
        with open(os.path.join("data", file), "r", encoding="utf-8") as f:
            article = json.loads(f.read())
        key, outlines = article["key"], article["outlines"]
        if given_outline: gen_text = writer(key, outlines)
        else: gen_text = writer(key)
        if count_tokens(gen_text) < token:
            filename.append(file)
            prompts.append(gen_text)
    return filename, prompts, []

def topk(level, num, token, given_outline=False): # use top-5 reference passage
    writer = Writer(prompt_writer_given_outline) if given_outline else Writer(prompt_writer)
    filename, prompts, retrieves = [], [], []
    log = open(os.path.join("baseline", f"{level}_{num}_fail.txt"), "w")
    for file in tqdm(os.listdir("data")):
        if os.path.exists(os.path.join("baseline", f"{level}_{num}", file)): continue
        with open(os.path.join("data", file), "r", encoding="utf-8") as f:
            article = json.loads(f.read())
        key, outlines, refs = article["key"], article["outlines"], article["reference"]
        retrieve = get_top_related_docs(key, refs, num, level)
        refstr = [f"Document [{i}] (Title: {title}, {url}) {doc}" for i, (title, url, doc) in enumerate(retrieve, 1)]
        if given_outline: gen_text = writer(key, outlines, refstr)
        else: gen_text = writer(key, refs=refstr)
        if count_tokens(gen_text) < token:
            filename.append(file)
            prompts.append(gen_text)
            retrieves.append(retrieve)
        else:
            print(file, file=log)
    log.close()
    return filename, prompts, retrieves

def topk_preretrieve(level, num, token, given_outline=False):
    with open(f"data/retrieve/top-{level}-text.json") as f:
        preretrived = json.loads(f.read())
    writer = Writer(prompt_writer_given_outline) if given_outline else Writer(prompt_writer)
    filename, prompts, retrieves = [], [], []
    log = open(os.path.join("baseline", f"{level}_{num}_fail.txt"), "w")
    for item in tqdm(preretrived):
        fileid, chunks = item["id"], item["top-chunks"]
        file = f"{fileid}.json"
        if os.path.exists(os.path.join("baseline", f"{level}_{num}", file)): continue
        with open(os.path.join("data", file), "r", encoding="utf-8") as f:
            article = json.loads(f.read())
        key, outlines = article["key"], article["outlines"]
        retrieve = [re.sub(r"[^\x00-\x7f]+", "", chunk) for chunk in chunks][:num]
        refstr = [f"Document [{i}] {doc}" for i, doc in enumerate(retrieve, 1)]
        if given_outline: gen_text = writer(key, outlines, refstr)
        else: gen_text = writer(key, refs=refstr)
        if count_tokens(gen_text) < token:
            filename.append(file)
            prompts.append(gen_text)
            retrieves.append(retrieve)
        else:
            print(file, file=log)
    log.close()
    return filename, prompts, retrieves

if __name__ == "__main__":
    generate_level = {"vanilla": vanilla, "tfidf": topk, "bm25": topk}
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--level", type=str, default="dpr", help="retrieve level: vanilla(no retrieve), tfidf, bm25, dpr, gtr")
    parser.add_argument("-n", "--num", type=int, default=5, help="numbers of retrieve chunks")
    parser.add_argument("-m", "--model", default=False, action="store_true", help="use 16k gpt model")
    parser.add_argument("-o", "--outline", default=False, action="store_true", help="provide outline")
    parser.add_argument("-t", "--token", type=int, default=800, help="max tokens")
    args = parser.parse_args()
    model = "gpt-3.5-turbo-16k-0613" if args.model else "gpt-3.5-turbo-0613"
    if not os.path.exists("baseline"): os.mkdir("baseline")
    savedir = f"baseline/{args.level}_{args.num}"
    if not os.path.exists(savedir): os.mkdir(savedir)
    if args.level in generate_level.keys():
        files, prompts, refs = generate_level[args.level](args.level, args.full, args.num, max_tokens[model] - args.token, args.outline)
    else:
        files, prompts, refs = topk_preretrieve(args.level, args.full, args.num, max_tokens[model] - args.token, args.outline)
    results = chatgpts(prompts, model, temperature=0, max_tokens=args.token)
    for file, text, ref in zip(files, results, refs):
        with open(os.path.join("data", file), "r", encoding="utf-8") as f:
            article = json.loads(f.read())
        key, outlines = article["key"], article["outlines"]
        with open(os.path.join(savedir, file), "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "outline": outlines, "text": text, "retrieve": ref}, ensure_ascii=False))
