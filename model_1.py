import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score

class back_channel_sentence_1(torch.nn.Module):
    def __init__(self, len_block:int, batch_size:int):
        super(back_channel_sentence_1, self).__init__()
        self.len_block = 2*len_block
        self.batch_size = batch_size
        self.lstm = torch.nn.LSTM(
            input_size =  768,
            hidden_size = 128,
            num_layers = 2,
            batch_first= True
        )
        self.full_connection = torch.nn.Linear(self.len_block*128, 2)

    def forward(self, bert_feature):
        x = self.lstm(bert_feature)
        x = x[0]
        # print(x.size())
        x = x.reshape(self.batch_size, self.len_block*128)
        # print(x.size())
        out = self.full_connection(x)
        return out


def train_model(model:back_channel_sentence_1, data_loader):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)
    epoch = 1

    model.train()
    for num in range(epoch):
        result = []
        true = []
        amount_all = 0
        amount_correct = 0
        for index, real_data in enumerate(data_loader):
            bert_feature = real_data['bert_feature']
            target = real_data['target']
            out = model(bert_feature)
            loss = loss_function(out, target)
            max_value, out_index = torch.max(out, dim=1)
            result.append(out_index)
            true.append(target)
            n_correct = (out_index == target).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            amount_correct += n_correct
            amount_all += len(target)
            print('\033[1;32;46m' +str(n_correct / len(target)) +'\033[0m')
            print(n_correct / len(target))
            with open('result.txt', 'a+') as file:
                file.write(str(n_correct / len(target)) + '\n')
        print('\033[1;32;46m' +str(amount_correct / amount_all) +'\033[0m')
        print(amount_correct / amount_all)
        with open('result.txt', 'a+') as file:
            file.write('----------------------'+
                       str(amount_correct / amount_all) +
                       '-------------------'+'\n')

def valid_model(model:back_channel_sentence_1, data_loader):
    model.eval()






