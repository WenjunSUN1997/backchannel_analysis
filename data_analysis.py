import pandas as pd
import re

def get_num(sentence:str):
    return re.findall(r"\d+", sentence)

def get_index_sentence(mainchannel:pd.DataFrame):
    '''输入语句列表，返回语句编号和纯净语句和speaker'''
    index_sentence_speaker = {}
    for sentence_block in mainchannel:
        if not pd.isna(sentence_block):
            for sentence in sentence_block.split('\n'):
                try:
                    index  = int(get_num(sentence)[0])
                    sentence_pure = sentence[sentence.find(':')+1:].replace('\xa0', '')
                    if  'Ps:' in sentence:
                        speaker = 'd'
                    else:
                        speaker = 'p'
                    index_sentence_speaker[index] = {'sentence': sentence_pure, 'speaker':speaker}
                except:
                    continue

    return index_sentence_speaker

def back_channel_process(backchannel:pd.DataFrame):
    '''输入语句列表，返回语句编号，纯净语句和speaker'''
    backchannel_with_index_type = {}
    for sentence_block in backchannel:
        print(sentence_block)
        if not pd.isna(sentence_block):
            for sentence in sentence_block.split('\n'):
                try:
                    index = int(get_num(sentence)[0])
                    if 'Ps:' in sentence:
                        speaker = 'd'
                    else:
                        speaker = 'p'
                    sentence_pure = sentence[sentence.find(':')+1:].replace('\xa0', '')
                except:
                    continue
                backchannel_with_index_type[index] = {'sentence':sentence_pure, 'speaker':speaker}
    return backchannel_with_index_type

def create_dataset(main_channel:dict, back_channel:dict):
    sentence_num = [1, 2, 3]

    for back_channel_index, back_channel_sentence in back_channel.items():
        for num in sentence_num:
            sentence_block = []
            left_num = num
            index_to_find = back_channel_index-1
            while left_num != 0 and index_to_find >= 0:
                if main_channel.get(index_to_find):
                    sentence_block.insert(0,main_channel[index_to_find]['sentence']+'_'
                                          +main_channel[index_to_find]['speaker'])
                    left_num -= 1
                index_to_find -= 1

            left_num = num
            index_to_find = back_channel_index+1
            max_main_channel_index = sorted(main_channel.keys())[-1]
            while left_num != 0 and index_to_find <= max_main_channel_index:
                if main_channel.get(index_to_find):
                    sentence_block.append(main_channel[index_to_find]['sentence']+'_'
                                          +main_channel[index_to_find]['speaker'])
                    left_num -= 1
                index_to_find += 1
            back_channel[back_channel_index][num] = sentence_block

    return back_channel

def create_fake(main_channel:dict):
    '''创建假数据'''
    max_main_channel_index = sorted(main_channel.keys())[-1]
    num_block = [1, 2]
    result = []

    for index_main_channel in main_channel.keys():
        if main_channel.get(index_main_channel+1):
            temp_1 = [main_channel[index_main_channel]['sentence']+'_'
                    +main_channel[index_main_channel]['speaker'],
                    main_channel[index_main_channel+1]['sentence']+'_'
                    +main_channel[index_main_channel+1]['speaker']]
            result.append(temp_1)
            for num in num_block:
                num_left = num
                temp_2 = [main_channel[index_main_channel]['sentence'] + '_'
                          + main_channel[index_main_channel]['speaker'],
                          main_channel[index_main_channel + 1]['sentence'] + '_'
                          + main_channel[index_main_channel + 1]['speaker']]
                # 向上寻找
                index_to_find = index_main_channel - 1
                while index_to_find >=0 and num_left != 0:
                    if main_channel.get(index_to_find):
                        temp_2.insert(0, main_channel[index_to_find]['sentence']+'_'+
                                    main_channel[index_to_find]['speaker'])
                        num_left -= 1
                    index_to_find -= 1
                #向下寻找
                num_left = num
                index_to_find = index_main_channel + 2
                while index_to_find <= max_main_channel_index and num_left != 0:
                    if main_channel.get(index_to_find):
                        temp_2.append(main_channel[index_to_find]['sentence']+'_'+
                                    main_channel[index_to_find]['speaker'])
                        num_left -= 1
                    index_to_find += 1
                if len(temp_2) == 2 * num + 2:
                    result.append(temp_2)
                temp = [main_channel[index_main_channel]['sentence'] + '_'
                        + main_channel[index_main_channel]['speaker'],
                        main_channel[index_main_channel + 1]['sentence'] + '_'
                        + main_channel[index_main_channel+1]['speaker']]

    return result

def storage_data(file_name:str, fake_data:list, back_channel:pd.DataFrame):
    sentence = []
    speaker = []
    block_1 = []
    block_2 = []
    block_3 = []
    block_fake_1 = []
    block_fake_2 = []
    block_fake_3 = []
    length = [len(sentence_block) for sentence_block in fake_data]
    for index in range(len(length)):
        if length[index] == 2:
            block_fake_1.append(fake_data[index])
        if length[index] == 4:
            block_fake_2.append(fake_data[index])
        if length[index] == 6:
            block_fake_3.append(fake_data[index])
    fake_dict = {'block_fake_1':block_fake_1,
                'block_fake_2':block_fake_2,
                'block_fake_3':block_fake_3}
    fake_data_csv = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in fake_dict.items()]))
    fake_data_csv.to_csv('data_processed/'+ file_name+'_fake_bc.csv', encoding='utf-8')

    for key, value in back_channel.items():
        sentence.append(value['sentence'])
        speaker.append(value['speaker'])
        block_1.append(value[1])
        block_2.append(value[2])
        block_3.append(value[3])
    back_channel_df = pd.DataFrame({'back_channel': sentence, 'speaker': speaker,
                                    'block_1': block_1, 'block_2': block_2, 'block_3': block_3})
    back_channel_df.to_csv('data_processed/'+ file_name + "_back_channel.csv")

def process(sheet_name):
    data_raw = pd.read_excel(io='data/Corpus.xlsx', sheet_name=sheet_name)

    data_to_process = data_raw[['"Main channel"', '"Back channel"']]
    main_channel = get_index_sentence(data_to_process['"Main channel"'])
    back_channel = back_channel_process(data_to_process['"Back channel"'])
    # print(main_channel)
    # print(back_channel)
    fake = create_fake(main_channel)
    back_channel = create_dataset(main_channel, back_channel)
    print(back_channel)
    storage_data(sheet_name, fake, back_channel)



if __name__ == '__main__':
    sheet_name_all = list(pd.read_excel(io='data/Corpus.xlsx', sheet_name=None).keys())
    sheet_name  = [x for x in sheet_name_all if '_Resu' not in x]
    print(sheet_name)
    print(len(sheet_name))
    for name in sheet_name:
        print(name)
        process(name)


