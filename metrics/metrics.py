import os
import json
import re
from tqdm import tqdm
from functools import partial
import nltk
from nltk import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import argparse
def bleu(ref: str, cand: str, n=4):
    weights = {1: (1, 0, 0, 0), 2: (0.5, 0.5, 0, 0), 3: (1 / 3, 1 / 3, 1 / 3, 0), 4: (0.25, 0.25, 0.25, 0.25)}
    ref, cand = ref.split(), cand.split()
    return sentence_bleu([ref], cand, weights=weights[n])
def meteor(ref: str, cand: str):
    nltk.download("wordnet")
    ref, cand = ref.split(), cand.split()
    return meteor_score([ref], cand)

def rouge_l(ref: str, cand: str):
    scorer = rouge.Rouge()
    return scorer.get_scores(cand, ref, avg=True)["rouge-l"]["f"]
def compute_metrics_2(ref: str, cand: str):
    metrics = {}
    scorers = {"BLEU_1": partial(bleu, n=1), "BLEU_4": partial(bleu, n=4), "METEOR": meteor, "ROUGE_L": rouge_l}
    for scorer in scorers:
        try:
            metrics[scorer] = scorers[scorer](ref, cand)
        except Exception as e:
            print(e)
            metrics[scorer] = 0
    return metrics
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--path",type=str)
    args=parser.parse_args()
    path=args.path
    files=os.listdir("../data/")
    len_sum=0
    dict1={}
    len_text=0
    keys=["BLEU_1","BLEU_4","METEOR","ROUGE_L"]
    for key in keys:
        dict1[key]=0
    for file in tqdm(files):
        with open("../data/"+file) as f:
            data_ref=json.load(f)
        if not os.path.exists(path+"/"+file):
            print("NOFILE",file)
            continue
        with open(path+"/"+file) as f:
            data_can=json.load(f)
        len_sum+=1
        if isinstance(data_can["text"],list):
            data_can["text"]="\n".join(data_can["text"])
        dict_tmp=compute_metrics_2(data_ref["text"],data_can["text"])
        len_text+=len(word_tokenize(data_can["text"]))
        for key in keys:
            dict1[key]+=dict_tmp[key]
    print(path)
    for key in keys:
        dict1[key]=dict1[key]/len_sum
        print(key+" is:",dict1[key]*100,"%")
    print("average length is:",len_text/len_sum)
if __name__=="__main__":
    main()