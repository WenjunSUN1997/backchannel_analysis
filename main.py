import torch
import create_dataset
import data_class
import model_1

batch_size = 8

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    with open('result.txt', 'w') as file:
        file.write(' ')

    data_true, data_fake = create_dataset.get_data(1)
    border = int(0.7*len(data_true))
    data_train = data_true[:border].append(data_fake[:border], ignore_index=True)
    data_train = data_class.data_loader(data_train)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    model = model_1.back_channel_sentence_1(1, batch_size)
    model.to(device)

    model_1.train_model(model, train_loader)
    torch.save(model.state_dict(), 'model_5_3')

