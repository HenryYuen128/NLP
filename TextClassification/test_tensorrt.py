import torch
from utils.DataProcess import load_dataset, TextClassficationDataset, SimpleDataset
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.TextClassification import BertFamily
import time
import torch_tensorrt

device = 'cuda'

df = pd.read_csv('/home/henry/code/NLP/data/ChnSentiCorp_htl_all.csv')

tokenizer = AutoTokenizer.from_pretrained('/home/henry/code/NLP/hotel_sentiment/hfl-chinese-roberta-wwm-ext_bert_BCE_mean_NO_2023-09-08_15:17:18/checkpoint-245')
encodings = tokenizer(df['review'].tolist(), max_length=128, truncation=True, padding='max_length', return_token_type_ids=True, return_tensors='pt')



test_dataset = SimpleDataset(encodings, df['review'].tolist())

dataloader = DataLoader(dataset=test_dataset, num_workers=0, shuffle=False,
                                                   batch_size=128)

model = torch.jit.load('/home/henry/code/NLP/trt_ts_module.ts')
model.to(device)
model.eval()

logits_list =list()

with torch.no_grad():
    start_time = time.time()
    for inputs in tqdm(dataloader, desc='Training...', position=0, leave=True):
        # labels = inputs['labels'].to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        # cate_feat = inputs['cate_feat'].to(device)

        logits = model(input_ids, token_type_ids, attention_mask)
        logits_list.append(logits)
    print(f'cost time(s): {time.time()-start_time}')

import pickle
with open('tensorrt_output.pkl', 'wb') as f:
    pickle.dump(torch.cat(logits_list, dim=0).cpu(), f)
