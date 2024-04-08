data_path = '' #path to ur data

import json 
import os 
def reverse_roles(data):
    for dic in data[1:]: # first will always be sys which we skip
        if dic['role'] == 'user':
            dic['role'] = 'assistant'
        elif dic['role'] == 'assistant':
            dic['role'] = 'user'         
    return data[1:] if data[-1]['role'] == 'assistant' else data[1:-1]

files = os.listdir(data_path)
#print(files)
dataset = []
for file in files:
    with open(f'{data_path}/{file}', 'r') as f:
        data = json.load(f)
    dataset.append({'messages':reverse_roles(data)})
#print(dataset)

with open(f'{data_path}/dataset.json', 'w') as f:
    json.dump(dataset, f)