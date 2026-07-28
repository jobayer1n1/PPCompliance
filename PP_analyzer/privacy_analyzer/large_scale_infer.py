from transformers import BertTokenizer, BertModel,BertConfig
import torch
import os
import sys
from pathlib import Path
from bs4 import BeautifulSoup
import csv

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent.parent / "NLP_models" / "BERT" / "bert_our_dataset"
sys.path.insert(0, str(TRAIN_DIR))
from train import CustomEffNet


def load_customeffnet_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Lightning saves LitPrivacy.model(CustomEffNet.model(...)) as
    # model.model.*, while plain CustomEffNet expects model.*.
    if state_dict and all(key.startswith('model.model.') for key in state_dict):
        state_dict = {
            key.replace('model.model.', 'model.', 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)

#split article into sentences
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = r"(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](jiwejfowejfojfojo|net|org|io|gov)"
def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub(r"\s" + alphabets + r"[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    res_sentences=[]
    for sentence in sentences:
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,.!#$%^&*()_+-=[]{}?/:;\'\"')
        sentence = ''.join(filter(whitelist.__contains__, sentence))
        if len(sentence)>500:
            sentence=sentence[:500]
        res_sentences.append(sentence)
    #print(sentences)
    return res_sentences


# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CustomEffNet()
model_path = SCRIPT_DIR / 'models' / 'epoch=34-valid_loss=1.1593-valid_acc=0.6981.ckpt'
load_customeffnet_state(model, model_path, device)
model.to(device)
model.eval()
embeddingmodel = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sm = torch.nn.Softmax(dim=-1)

# load sentences
# policy_folder = str(SCRIPT_DIR.parent / 'raw_data' / 'privacy_data')

policy_folder = str(SCRIPT_DIR / 'top10ext')

policy_conclude_predict=[]
policy_raw_result=[]
for root, dirs, files in os.walk(policy_folder):
    from tqdm import tqdm
    count=0
    for file in tqdm(files):
        # count+=1
        # if count>20:
        #     break
        
        ext_id=file[:-5]
        ext_result=[0]*21
        ext_result_top2=[0]*21
        # print(ext_result)
        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
            try:
                tmp = f.read()
            except:
                continue
            soup = BeautifulSoup(tmp, 'html.parser')
            text=soup.get_text()
            text=text.replace('  ', '')
            text=text.replace('\\n', '')
            text=text.replace('\\t', '')
            text=text.replace('\\', '')
            text=text.replace('xe2x80x9c', '') # “
            text=text.replace('xe2x80x9d', '') # ”
            text=text.replace('xe2x80x99', '') # ’
            text=text.replace('xe2x8098', '')  # ‘
            text=text.replace('xe2x86x92', '') # →
            text=text.replace('b\'', '')       # The leading b' that appears when converting a Python bytes object to a string via str(b_data)
            text=text.replace('xe2x80x8b', '') # Zero-width space
            text=text.replace('xc2xa0', '')    # Non-breaking space
            text=text.replace('xc2xb7', '')    # Middle dot / Bullet · 
            text=text.replace('rr', '')        
            sentences=split_into_sentences(text)
            for sentence in sentences:
                if len(sentence)>20:
                    a = tokenizer.encode(sentence, add_special_tokens=True)
                    embedding_res = embeddingmodel(torch.tensor(a).unsqueeze(0))[1].detach().to(device)

                    with torch.no_grad():
                        pred = model(embedding_res).squeeze()
                    pred = sm(pred)
                    pred = pred.detach().cpu().tolist()
                    # pred_res=pred.index(max(pred))
                    pred_res=sorted(pred)
                    pred_res1=pred.index(pred_res[-1])
                    pred_res2=pred.index(pred_res[-2])
                    ext_result[pred_res1]+=1
                    ext_result_top2[pred_res1]+=1
                    ext_result_top2[pred_res2]+=1
                    with open(SCRIPT_DIR / 'privacy_conclude_raw.csv','a') as f:
                        csvf=csv.writer(f)
                        res=[ext_id,sentence]+[str(i) for i in pred]
                        csvf.writerow(res)
                # print(ext_id,pred_res)
        with open(SCRIPT_DIR / 'privacy_conclude_result_top1.csv','a') as f:
            csvf=csv.writer(f)
            csvf.writerow([ext_id]+[str(i) for i in ext_result])
        with open(SCRIPT_DIR / 'privacy_conclude_result_top2.csv','a') as f:
            csvf=csv.writer(f)
            csvf.writerow([ext_id]+[str(i) for i in ext_result_top2])
        policy_conclude_predict.append([ext_id]+ext_result)

with open(SCRIPT_DIR / 'privacy_conclude_result_last.csv','w') as f:
    csvf=csv.writer(f)
    csvf.writerows(policy_conclude_predict)
'''        
sentence = 'This site provides any third party cookies and makes no effort to track you.'

a = tokenizer.encode(sentence, add_special_tokens=True)
embedding_res = embeddingmodel(torch.tensor(a).unsqueeze(0))[1].detach().to(device)

with torch.no_grad():
    pred = model(embedding_res).squeeze()
pred = sm(pred)

pred = pred.detach().cpu().tolist()
print
'''
