import random

import numpy as np
import pandas as pd
import json
import math
import os

#####config
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

name = "taiwan.csv"
feature_size = 95 + 1  # Target_index = 0
train_size, dev_size, test_size = 0.7, 0.1, 0.2

if not math.isclose(train_size + dev_size + test_size, 1.0):
    print("sample size wrong!!!")

os.makedirs('gpt4-data', exist_ok=True)

# Feature names are read straight from the CSV header (skipping the target column,
# "Bankrupt?", which is column 0) rather than hardcoded, so the labels used in the
# generated text can never drift out of sync with the actual data columns.
_data_raw = pd.read_csv(name, sep=',', header=0, skipinitialspace=True)
column_name = _data_raw.columns.values.tolist()
mean_list = column_name[1:]


#####function
def data_split(data):
    random.seed(10086)

    train_ind = random.sample([i for i in range(len(data))], int(len(data) * train_size))
    train_data = [data[i] for i in train_ind]

    index_left = list(set(list(range(len(data)))) - set(train_ind))
    dev__ind = random.sample(index_left, int(len(data) * dev_size))
    dev_data = [data[i] for i in dev__ind]

    index_left = list(set(index_left) - set(dev__ind))
    test_data = [data[i] for i in index_left]

    return train_data, dev_data, test_data


def process_table(data, mean_list):
    data_tmp = []
    prompt = "Predict whether the company will face bankruptcy based on the financial profile attributes provided in the following text. " \
             "Respond with only 'no' or 'yes', and do not provide any additional information. \n"
    from_text = "The client has attributes:  ROA(C) before interest and depreciation before interest: 0.499, " \
                "..., Net Income Flag: 1.000,  Equity to Liability: 0.044."
    prompt = prompt + f"For instance, '{from_text}' should be classified as 'no'. \nText: "
    for j in range(len(data)):
        text = 'The client has attributes: '
        for i in range(1, len(data[0])):
            sp = ', ' if i != len(data[0]) - 1 else '.'
            text = text + f'{mean_list[i - 1]}: {data[j][i]:.3f}' + sp
        answer = 'no' if data[j][0] == 0 else 'yes'
        # '0' is good (no bankruptcy) (6599) and '1' is bad
        data_tmp.append(
            {'id': j, "query": f"{prompt}'{text}'" + ' \nAnswer:', 'answer': answer, "choices": ["no", "yes"],
             "gold": int(data[j][0]), 'text': text})
    return data_tmp


def json_save(data, dataname, mean_list=mean_list, out_jsonl=True):
    data_tmp = process_table(data, mean_list)
    if out_jsonl:
        with open('{}.jsonl'.format(dataname), 'w') as f:
            for i in data_tmp:
                json.dump(i, f)
                f.write('\n')
            print('-----------')
            print(f"{dataname}.jsonl write done")
        f.close()
    # df = pd.DataFrame(data_tmp)
    # # 保存为 Parquet 文件
    # parquet_file_path = f'data/{dataname}.parquet'
    # df.to_parquet(parquet_file_path, index=False)
    return data_tmp


def json_save_gpt4(data, dataname, mean_list=mean_list):
    data_tmp = process_table(data, mean_list)
    with open(f'gpt4-data/{dataname}.jsonl', 'w') as f:
        for i in data_tmp:
            json.dump(i, f)
            f.write('\n')
        print('-----------')
        print(f"{dataname}.jsonl write done")
    f.close()


def save_gpt4_data(test_data):
    tmp_data = [row[0] for row in test_data]
    _, gpt4_data = train_test_split(test_data, test_size=500, stratify=tmp_data, random_state=100)

    s1_data = [row for row in gpt4_data if row[0] == 0]
    s2_data = [row for row in gpt4_data if row[0] == 1]
    s_data = s2_data + s1_data[:max(0, 100 - len(s2_data))]
    s_data = pd.DataFrame(s_data)
    np.random.seed(42)
    random_index = np.random.permutation(s_data.index)
    ss_data = s_data.reindex(random_index)
    # ss_data.to_csv('gpt4_rawdata.csv', index=False)
    json_save_gpt4(ss_data.values.tolist(), 'test_gpt4')


#####process
# column 0 is the target variable on data
data = _data_raw.values.tolist()

data = data_split(data)

save_gpt4_data(data[2])

save_name = ['train', 'valid', 'test']
for i in range(len(data)):
    _ = json_save(data[i], save_name[i])