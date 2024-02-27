import os, re, json
import argparse
from tqdm import tqdm
from prompt import *
from utils import max_tokens, count_tokens
from callgpt import chatgpts

class Rewriter(object):

    def __init__(self, prompt):
        self.prompt = prompt

    def __call__(self, src):
        user_prompt = f"Original Passage:\n{src}"
        return [{"role": "system", "content": self.prompt}, {"role": "user", "content": user_prompt}]

def rewrite(model, token):
    writer = Rewriter(prompt_rewrite)
    filename, prompts = [], []
    log = open(f"{model}_fail.txt", "w")
    for file in tqdm(os.listdir(f"baseline_2stage/{model}")):
        if os.path.exists(os.path.join(f"baseline_rewrite/{model}", file)): continue
        with open(os.path.join(f"baseline_2stage/{model}", file), "r", encoding="utf-8") as f:
            article = json.loads(f.read())
        text = article["key"], article["text"]
        gen_text = writer("\n".join(text))
        if count_tokens(gen_text) < token:
            filename.append(file)
            prompts.append(gen_text)
        else:
            print(file, file=log)
    log.close()
    return filename, prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=False, action="store_true", help="use 16k gpt model")
    parser.add_argument("-f", "--full", default=False, action="store_true", help="use full dataset")
    parser.add_argument("-t", "--token", type=int, default=4096, help="max tokens")
    args = parser.parse_args()
    if args.model:
        window = "16k"
        model = "gpt-3.5-turbo-16k-0613"
    else:
        window = "4k"
        model = "gpt-3.5-turbo-0613"
    if not os.path.exists("baseline_rewrite"): os.mkdir("baseline_rewrite")
    savedir = f"baseline_rewrite/{window}"
    if not os.path.exists(savedir): os.mkdir(savedir)
    files, prompts = rewrite(window, max_tokens[model] - args.token)
    results = chatgpts(prompts, model, temperature=0, max_tokens=args.token)
    for file, text in zip(files, results):
        with open(os.path.join(f"baseline_2stage/{window}", file), "r", encoding="utf-8") as f:
            article = json.loads(f.read())
        article["text"] = text
        with open(os.path.join(savedir, file), "w", encoding="utf-8") as f:
            f.write(json.dumps(article, ensure_ascii=False))
