import pandas as pd
from transformers import AutoTokenizer, BertModel
from torch import tensor
from torch.utils.data import Dataset
from ast import literal_eval
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class data_loader(Dataset):
    def __init__(self,data):
        self.data = data
        self.max_len = 256
        self.tokenizer = AutoTokenizer.from_pretrained('camembert-base')
        self.bert_model = BertModel.from_pretrained('camembert-base')

    def __len__(self):
        return len(self.data['sentence'])

    def __getitem__(self, item):
        print(item)
        sentence_list = literal_eval(self.data['sentence'][item])
        target = self.data['target'][item]
        print(sentence_list)
        output_tokenizer = self.tokenizer(sentence_list, max_length=self.max_len, padding='max_length')
        input_ids = tensor(output_tokenizer['input_ids'])
        attention_mask = tensor(output_tokenizer['attention_mask'])

        bert_feature = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return {'bert_feature': bert_feature['last_hidden_state'][:, :1, :].squeeze().to(device),
                'target': tensor(target).to(device)}



