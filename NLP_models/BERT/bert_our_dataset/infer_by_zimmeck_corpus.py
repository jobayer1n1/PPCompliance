from train import CustomEffNet,LitPrivacy
from transformers import BertTokenizer, BertModel,BertConfig
import torch
import os
from bs4 import BeautifulSoup
from tqdm import tqdm
import csv
import pandas as pd
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


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

def match_after_label_mapping(pred,target):
    # uncommen if test is other set
    pred,target=target,pred
    
    # from zimmeck to our corpus
    if pred==0 and target==0:
        return True
    if pred!=0 and target==1:
        return True
    if pred>=1 and pred<=16 and target==14:
        return True
    if pred>=17 and target<=22 and target==1:
        return True
    if pred>=23 and pred<=46 and target==13:
        return True
    if pred>=47 and pred<=58 and target==15:
        return True
    return False

    # # from liushaung to our corpus
    # if pred==target:
    #     return True
    # if pred==1 and target>=11 and target<=19:
    #     return True
    # if pred==0 and target==20:
    #     return True
    # return False

    

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CustomEffNet()
model_path = SCRIPT_DIR / 'logs' / 'epoch=41-valid_loss=1.0552-valid_acc=0.7201.ckpt'
load_customeffnet_state(model, model_path, device)
model.to(device)
model.eval()
embeddingmodel = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sm = torch.nn.Softmax(dim=-1)

# load sentences
policy_file=SCRIPT_DIR / 'zimmeck_dataset_filtered.csv'
col_list=['label','review','filename']
df=pd.read_csv(policy_file, usecols=col_list)

sentences=df['review']
target=df['label']
correct=0
for sentence in tqdm(sentences):
    a = tokenizer.encode(sentence, add_special_tokens=True)
    embedding_res = embeddingmodel(torch.tensor(a).unsqueeze(0))[1].detach().to(device)

    with torch.no_grad():
        pred = model(embedding_res).squeeze()
    pred = sm(pred)
    pred = pred.detach().cpu().tolist()
    # pred_res=pred.index(max(pred))
    pred_res=sorted(pred)
    pred_res1=pred.index(pred_res[-1])
    target_res=target[df.index[df['review'] == sentence].tolist()[0]]
    if match_after_label_mapping(pred_res1,target_res):
        correct += 1
    # print(pred_res1,target_res)
print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(sentences), 100. * correct / len(sentences)) )
