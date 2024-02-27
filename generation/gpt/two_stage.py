import os, re, json
import argparse
from tqdm import tqdm
from prompt import *
from utils import max_tokens, count_tokens
from callgpt import chatgpts

class Writer(object):
    def __init__(self, prompt):
        self.prompt = prompt

    def __call__(self, key, subtitle=None, refs=None):
        user_prompt = f"Keyword: {key}\n"
        if subtitle:
            if isinstance(subtitle, str): user_prompt += f"Section: {subtitle}\n"
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

def gen_outline(level, num):
    with open(f"data/retrieve/top-{level}-text.json") as f:
        preretrived = json.loads(f.read())
    writer = Writer(prompt_outline)
    filename, prompts, retrieves = [], [], []
    for item in tqdm(preretrived):
        id, chunks = item["id"], item["top-chunks"]
        file = f"{id}.json"
        if os.path.exists(os.path.join("baseline", f"{level}_{num}", file)): continue
        with open(os.path.join("data", file), "r", encoding="utf-8") as f:
            article = json.loads(f.read())
        key = article["key"]
        retrieve = [re.sub(r"[^\x00-\x7f]+", "", chunk) for chunk in chunks][:num]
        refstr = [f"Document [{i}] {doc}" for i, doc in enumerate(retrieve, 1)]
        gen_text = writer(key, refs=refstr)
        filename.append(file)
        prompts.append(gen_text)
        retrieves.append(retrieve)
    return filename, prompts, retrieves

def per_section(level, num, token):
    with open("data/retrieve/top-dpr-gpt-per-section-text.json") as f:
        preretrived = json.loads(f.read())
    writer = Writer(prompt_writer_per_section)
    filename, prompts, retrieves = [], [], []
    log = open(os.path.join("baseline", f"{level}_{num}_fail.txt"), "w")
    for item in tqdm(preretrived):
        fileid, sections, allchunks = item["id"], item["sections"], item["top-chunks"]
        file = f"{fileid}.json"
        if os.path.exists(os.path.join("baseline", f"{level}_{num}", file)): continue
        with open(os.path.join("data", file), "r", encoding="utf-8") as f:
            article = json.loads(f.read())
        key = article["key"]
        for subtitle, chunks in zip(sections, allchunks):
            retrieve = [re.sub(r"[^\x00-\x7f]+", "", chunk) for chunk in chunks][:num]
            refstr = [f"Document [{i}] {doc}" for i, doc in enumerate(retrieve, 1)]
            gen_text = writer(key, subtitle, refstr)
            if count_tokens(gen_text) < token:
                filename.append((fileid, subtitle))
                prompts.append(gen_text)
                retrieves.append(retrieve)
            else:
                print(file, subtitle, file=log)
    log.close()
    return filename, prompts, retrieves

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--level", type=str, default="dpr", help="retrieve level: vanilla(no retrieve), tfidf, bm25, dpr, gtr")
    parser.add_argument("-n", "--num", type=int, default=5, help="numbers of retrieve chunks")
    parser.add_argument("-m", "--model", default=False, action="store_true", help="use 16k gpt model")
    parser.add_argument("-o", "--outline", default=False, action="store_true", help="provide outline")
    parser.add_argument("-t", "--token", type=int, default=400, help="max tokens")
    parser.add_argument("--outline_only", default=False, action="store_true", help="only generate outline")
    args = parser.parse_args()
    model = "gpt-3.5-turbo-16k-0613" if args.model else "gpt-3.5-turbo-0613"
    if not os.path.exists("baseline"): os.mkdir("baseline")
    savedir = f"baseline/{args.level}_{args.num}"
    if not os.path.exists(savedir): os.mkdir(savedir)
    if args.outline_only: files, prompts, refs = gen_outline(args.level, args.full, args.num)
    else: files, prompts, refs = per_section(args.level, args.num, args.src, max_tokens[model] - args.token)
    results = chatgpts(prompts, model, temperature=0, max_tokens=args.token)
    fileid, fullfile = -1, {}
    for (current_fileid, section), text, ref in zip(files, results, refs):
        if fileid != current_fileid:
            if fileid >= 0:
                with open(os.path.join(savedir, f"{fileid}.json"), "w", encoding="utf-8") as f:
                    f.write(json.dumps(fullfile, ensure_ascii=False))
            fileid = current_fileid
            with open(os.path.join("data", f"{fileid}.json"), "r", encoding="utf-8") as f:
                article = json.loads(f.read())
            fullfile = {"key": article["key"], "outline": article["outlines"], "text": [], "retrieve": []}
        fullfile["text"].append(text)
        fullfile["retrieve"].append(ref)
    if fileid >= 0:
        with open(os.path.join(savedir, f"{fileid}.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(fullfile, ensure_ascii=False))
