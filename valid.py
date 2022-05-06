import torch
import sys
import model_1
import data_class
import create_dataset

batch_size = 8

if __name__ =="__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    len_block = int(sys.argv[1])
    model_path = 'model/model_5_4'+sys.argv[1]
    model_test = model_1.back_channel_sentence_1(len_block, batch_size)
    model_test.load_state_dict(torch.load(model_path))
    model_test.to(device)

    with open('model_5_4'+str(2*len_block)+'test.txt', 'w') as file:
        file.write(' ')

    data_true, data_fake = create_dataset.get_data(len_block)
    border = int(0.7 * len(data_true))
    data_test = data_true[border:].append(data_fake[border:], ignore_index=True)
    data_test = data_class.data_loader(data_test)
    print(len(data_test))
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

    model_1.valid_model(model_test, test_loader)




