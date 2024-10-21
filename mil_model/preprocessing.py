import pandas as pd
import numpy as np 
import json

def load_labels(file_path):

    df_labels = pd.read_csv(file_path, delimiter=',')

    labels = df_labels['label'].tolist()
    labels = list(df_labels['label'])

    gene_id = list(df_labels['gene_id'])
    transcript_id = list(df_labels['transcript_id'])
    transcript_pos = list(df_labels['transcript_position'])

    return labels, gene_id, transcript_id, transcript_pos

def load_dataset_json(file_path):

    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line) 
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}")
                print(f"Error message: {e}")

    return data_list

def build_bags(data_list):

    bags = [] 
    for dat in (data_list):
        for _, pos in dat.items():  
            for _, seq in pos.items():  
                for _, measurements in seq.items(): 
                    bag = []
                    for read in measurements:
                        instance = np.array(read).reshape(9,) 
                        bag.append(instance)
                    if len(bag) == 0:
                        print(bag) 
                    bags.append(bag)
    
    return bags
