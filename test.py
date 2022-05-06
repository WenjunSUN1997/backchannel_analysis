import pandas as pd
from ast import literal_eval
from transformers import  AutoTokenizer, BertModel
import torch
import numpy as np
import gc
import data_class
import model_1
import os

# tokenizer = AutoTokenizer.from_pretrained('camembert-base')
# model = BertModel.from_pretrained('camembert-base')
# sentence_list = ["Donc euh... en réalité on gagne pas trois euros par jour si on fait le calcul ça fait soixante euros par mois. mais en réalité on gagne pas que ça hein parce qu'on touche l'AAH donc 776..._p", "on a l'APL 350 plus les 60 du travail ça fait à peu près 1200 euros par mois quoi._p"]
# output = tokenizer(sentence_list, max_length = 128, padding = 'max_length')
# print(output)
# print('te')
# feature = model(input_ids = torch.tensor(output['input_ids']), attention_mask = torch.tensor(output['attention_mask']))
# print(feature[0][:, :1].size())
# a = [[[1,2],[3,4]],[[5,6],[7,8]]]
# a= torch.tensor(a)
# print(a[:, :1])







# path = 'data_processed/001PEPout_back_channel.csv'
# df_temp = pd.read_csv(path)['block_3'].dropna()
# df = {'sentence':df_temp,
# 	  'target':[1] * len(df_temp)}
#
# data_test = data_class.data_loader(df)
# print(data_test.__len__())
# train_loader = torch.utils.data.DataLoader(data_test, batch_size=4, shuffle=True)
#
#
# model = model_1.back_channel_sentence_1(6)
#
# model_1.train_model(model,train_loader)
# file_list_all = os.listdir('data_processed')
# file_list = [x for x in file_list_all if 'T'==x[0]]
#
# file_true = [x for x in file_list if 'fake' not in x]
# file_fake = [x for x in file_list if 'fake' in x]
#
#
#
# for file_name in file_true:
#     try:
#         df = pd.read_csv('data_processed/'+file_name)['block_3'].dropna()
#
#         for x in range(len(df)):
#             try:
#                 if len(literal_eval(df[x])) !=6:
#                     print(x)
#                     print(df[x])
#                     print(len(literal_eval(df[x])))
#                     print(file_name)
#             except:
#                 print(file_name)
#                 print(x)
#     except:
#         print(file_name)
#
# for file_name in file_fake:
#     df = pd.read_csv('data_processed/'+file_name)['block_fake_3'].dropna()
#     for x in range(len(df)):
#         if len(literal_eval(df[x])) !=6:
#             print(x)
#             print(df[x])
#             print(len(literal_eval(df[x])))
#             print(file_name)

x = torch.rand(4,2,2)
d =torch.nn.Dropout(0.5)
print(d(x).size())


