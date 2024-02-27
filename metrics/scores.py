import json
import os
from tqdm import tqdm
import argparse
import re
from openai import OpenAI
api_key = os.environ["OPENAI_API_KEY"]
base_url = os.environ["OPENAI_API_BASE"] if os.getenv("OPENAI_API_BASE") else "https://api.openai.com/v1/"
client = OpenAI(api_key=api_key, base_url=base_url)
def get_score(str1):
    numbers=re.findall(r"\d+",str1)
    scores=[]
    for number in numbers[:3]:
        try:
            num=int(number)
            if num>5 or num<0:
                scores.append(0)
            else:
                scores.append(num)    
        except:
            scores.append(0)
    if len(scores)!=3:
        print("ERROE not 3 scores!")
        scores=[0,0,0]
    return scores
def get_completion(prompt):
    messages = [{"role": "system", "content": prompt[0]}, {"role": "user", "content": prompt[1]}]
    model="gpt-4-1106-preview"
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content
prompt_system = "Evaluate a encyclopedia text of a keyword in three metrics,fluency,informativeness,faithfulness.Give a score from 0-5 to each metric. \nFluency: Assess the text for grammatical correctness, coherence of ideas, and overall readability. Look for smooth transitions between sentences and paragraphs, as well as clear organization of information.\nInformativeness: Evaluate the depth and breadth of information provided about the keyword. Check if the text covers various aspects of the topic, including its definition, background, significance, related concepts, and any relevant examples or applications.\nFaithfulness: Verify the accuracy of the information presented in the text by cross-referencing with credible sources or established knowledge,assess whether the information aligns with accepted facts and evidence.\nOnly give three scores in this form: Fluency:score1,Informativeness:score2 Faithfulness:socre3.No need to Explaination."
prompt_user="Keyword:{KEYWORD}\n\
    Encyclopedia Text:{WIKITEXT}\n\
    Scores:"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="")
    args=parser.parse_args()
    path=args.path
    files=os.listdir("../data/")
    method=os.path.basename(path)
    sfl=0
    sin=0
    sfa=0
    lenfiles=0
    for file in tqdm(files):
        with open("../data/"+file) as f:
            data_ref=json.load(f)
        if not os.path.exists(path+"/"+file):
            print("NOFILE",file)
            continue
        lenfiles+=1
        with open(path+"/"+file) as f:
            data_can=json.load(f)
        keyword=data_ref["key"]
        if isinstance(data_can["text"],list):
            data_can["text"]="\n".join(data_can["text"])
        prompt_u=prompt_user.replace("{KEYWORD}",keyword).replace("{WIKITEXT}",data_can["text"])
        try:
            ret=get_completion((prompt_system,prompt_u))
            score=get_score(ret)
        except Exception as e:
            print(e)
            score=[0,0,0]
            ret=""
        sfl+=score[0]
        sin+=score[1]
        sfa+=score[2]
        dict1={"keyword":keyword,"score":score,"gpt_reply":ret}
        if not os.path.exists("./test_scores/"+method):
            os.makedirs("./test_scores/"+method)
            print("mkdir")
        with open("./test_scores/"+method+"/"+file,"w") as f:
            json.dump(dict1,f)
    print(path)
    print(sfl/lenfiles)
    print(sin/lenfiles)
    print(sfa/lenfiles)
if __name__=="__main__":
    main()