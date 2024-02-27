import os
import json
import re
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
import argparse
device="cuda:0" if torch.cuda.is_available() else "cpu"
model_path = 'google/t5_11b_trueteacher_and_anli'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path,device_map=device)
model.eval()
def get_nli_predictions(premises:list,hypothesises:list):
    list1=[]
    decoded_outputs=[]
    for premise,hypothesis in zip(premises,hypothesises):
        list1.append(f'premise: {premise} hypothesis: {hypothesis}')
    for i in range(0,len(list1),8):
        encodings = tokenizer(list1[i:i+8], truncation=True, padding="longest",max_length=512, return_tensors='pt')
        encodings = {key: value.to(device) for key, value in encodings.items()}
        outputs = model.generate(**encodings)
    # 解码输出并打印结果
        decoded_outputs.extend([tokenizer.decode(label, skip_special_tokens=True) for label in outputs])
    return decoded_outputs
def split_and_merge(text):
    segments = re.split(r'(\[\d+\])', text)
    result = []
    current_sentence = ""
    for segment in segments:
        if re.match(r'\[\d+\]', segment):
            current_sentence += segment
        elif segment == "\n" or segment == "":
            current_sentence+=segment
        else:
            if current_sentence:
                result.append(current_sentence)
            current_sentence = segment
    if current_sentence:
        result.append(current_sentence)
    return result
def find_num(text):
    seg=""
    num=[]
    matches = re.findall(r'\[(\d+)\]', text)
    text = re.split(r'\[\d+\]', text)
    numbers = [int(match)-1 for match in matches]
    for t in text:
        seg+=t
    return seg,numbers
def get_citation_predictions(path):
    pres=0
    recalls=0
    files=os.listdir(path)
    for file in files:
        with open(path+file,"r",encoding="utf-8") as f:
            data=json.load(f)
            nlis=data["nli"]
            lsent=len(nlis)
            pre=0
            recall=0
            for nli in nlis:
                if len(nli)==0:
                    continue
                sum=0
                flag=0
                for i in nli:
                    if i=="1":
                        flag=1
                    if i=="0":
                        sum+=1
                pre+=1-sum/len(nli)
                recall+=flag
            if lsent!=0:
                pres+=pre/lsent
                recalls+=recall/lsent
    print(path)
    print("(hy that can be premise)Citation recall is:\n",recalls/len(files))
    print("(premise that can truely nli)Citation precision is:\n",pres/len(files))
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--path",type=str)
    args=parser.parse_args()
    path=args.path
    files=os.listdir("../data/")
    get_dirname=os.path.basename(path)
    test_path="./test_nli/"+get_dirname+"/"
    if not os.path.exists(test_path):
        os.mkdir(test_path)
        print("mkdir:"+test_path)
    for file in tqdm(files):
        if not os.path.exists(path+"/"+file):
            continue
        with open(path+"/"+file,"r",encoding="utf-8") as f:
            data=json.load(f)
            texts=data["text"]
            refers=data["retrieve"]
            if isinstance(texts,str):
                texts=split_and_merge(texts)
                text_refer=[find_num(text) for text in texts]
                premises=[]
                hypothesises=[]
                for i in text_refer:
                    for j in i[1]:
                        hypothesises.append(i[0])
                        try:
                            premises.append(refers[j])
                        except Exception as e:
                            #print(e)
                            premises.append(" ")
                nli_=get_nli_predictions(premises,hypothesises)
            elif isinstance(texts,list):
                text_refer=[]
                sum=0
                premises=[]
                hypothesises=[]
                for text,refer in zip(texts,refers):
                    text=split_and_merge(text)
                    text_ref=[find_num(tex) for tex in text]
                    text_refer.extend(text_ref)
                    for i in text_ref:
                        for j in i[1]:
                            hypothesises.append(i[0])
                            try:
                                premises.append(refer[j])
                            except Exception as e:
                                #print(e)
                                premises.append(" ")
                nli_=get_nli_predictions(premises,hypothesises)
            with open(test_path+file,"w",encoding="utf-8") as fw:
                list_nli=[]
                sum=0
                for i in range(len(text_refer)):
                    tmp=[]
                    tmp.extend(nli_[sum:sum+len(text_refer[i][1])])
                    sum+=len(text_refer[i][1])
                    list_nli.append(tmp)
                fw.write(json.dumps({"text_refer":text_refer,"nli":list_nli},ensure_ascii=False))
    get_citation_predictions(test_path)
    print(path)
if __name__ == '__main__':
    main()