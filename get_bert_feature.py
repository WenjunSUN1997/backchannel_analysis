import torch
from transformers import AutoTokenizer, BertModel
import pandas as pd
from ast import literal_eval
import os
import gc

max_length = 256

def initial(model_name:str):
    frech_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)
    return frech_tokenizer, bert_model

def get_feature(tokenizer:AutoTokenizer, model:BertModel, sentence_list:list):
    output_tokenizer = tokenizer(sentence_list, max_length=max_length, padding='max_length')
    input_ids = torch.tensor(output_tokenizer['input_ids'])
    attention_mask = torch.tensor(output_tokenizer['attention_mask'])
    features = model(input_ids=input_ids, attention_mask=attention_mask)[0][:, :1]

    return features


def storage_fake_feature(file_name:str):
    data = pd.read_csv('data_processed/' + file_name)
    amount = len(data)
    col_name = ['block_fake_1', 'block_fake_2', 'block_fake_3']

    for col in col_name:
        tokenizer, model = initial(model_name)
        num_sentence = int(col[-1])
        print(num_sentence)
        feature_temp = torch.zeros(amount, 2*num_sentence, max_length, 768)
        sentence_list_all = data[col]
        index = 0
        for sentence_list in sentence_list_all:
            if pd.isna(sentence_list):
                continue
            sentence_list_temp = literal_eval(sentence_list_all[index])
            print(sentence_list_temp)
            features = get_feature(tokenizer, model, sentence_list_temp)
            feature_temp[index] = features
            index += 1

        torch.save(feature_temp, 'feature/' + file_name + 'feature_fake_'+ col +'.pt')
        del feature_temp
        gc.collect()

def storage_bc_feature(file_name:str):
    data = pd.read_csv('data_processed/' + file_name)
    amount = len(data)
    col_name = ['block_1', 'block_2', 'block_3']

    for col in col_name:
        tokenizer, model = initial(model_name)
        num_sentence = 2 * int(col[-1])
        feature_temp = torch.zeros(amount, num_sentence, max_length, 768)
        index = 0
        sentence_list_all = data[col]
        for sentence_list in sentence_list_all:
            print(sentence_list)
            sentence_list_temp = literal_eval(sentence_list)
            features = get_feature(tokenizer, model, sentence_list_temp)
            feature_temp[index] = features
            index += 1

        torch.save(feature_temp, 'feature/' + file_name + 'feature_' + col + '.pt')
        del feature_temp
        gc.collect()

if __name__ == "__main__":
    model_name = 'camembert-base'
    file_name_list = os.listdir('data_processed')
    print(file_name_list)
    for file_name in file_name_list:
        if 'fake' in file_name:
            storage_fake_feature(file_name)
        if 'fake' not in file_name:
            storage_bc_feature(file_name)

    # file_name = '001PEPout_back_channel.csv'
    # storage_bc_feature(file_name, tokenizer, model)



