import torch
import create_dataset
import data_class
import model_1
import model_2
import sys

batch_size = 8

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    len_block = int(sys.argv[1])
    with open('result_5_5_'+str(2*len_block)+'.txt', 'w') as file:
        file.write(' ')

    data_true, data_fake = create_dataset.get_data(len_block)
    border = int(0.7*len(data_true))
    data_train = data_true[:border].append(data_fake[:border], ignore_index=True)
    data_train = data_class.data_loader(data_train)
    print(len(data_train))

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    model = model_2.back_channel_sentence_2(len_block, batch_size)
    model.to(device)
    model_2.train_model(model, train_loader)

    torch.save(model.state_dict(), 'model_5_5'+sys.argv[1])

