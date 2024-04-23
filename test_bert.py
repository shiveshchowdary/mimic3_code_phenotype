import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import math
import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
pd.set_option('future.no_silent_downcasting',True)
from normalizer import Normalizer
import json
from collections import defaultdict
CLASS_TOKEN = False
import json
with open("IdNameDict.json","r") as f:
    var_dict = json.load(f)

class MimicDataSetPhenotyping(Dataset):
    def __init__(self, data_dir, csv_file, mean_variance , mode, seq_len, uniform , var_dict = var_dict, pad_value = 0, device = DEVICE):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.seq_len = seq_len
        self.mode = mode
        self.data_df = csv_file
        self.data_df.replace(["21607_episode1_timeseries.csv"], np.nan, inplace=True)
        self.data_df.dropna(inplace=True) 
        self.data_df.reset_index(drop = True, inplace=True)
        self.mean_variance = mean_variance
        self.pad_value = pad_value
        self.device = device
        self.id_name_dict = {v:k for k,v in (var_dict.items())}
        self.uniform = uniform
            
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        path = self.data_dir + self.data_df['stay'][idx]
        data = pd.read_csv(path)
        # categorical_variables = ['Glascow coma scale eye opening', 
        #                          'Glascow coma scale motor response', 
        #                          'Glascow coma scale verbal response']
        id_name_dict = {}
        # data.drop(labels=categorical_variables, axis=1, inplace=True)
        data.replace(['ERROR','no data','.','-','/','VERIFIED','CLOTTED',"*",'ERROR DISREGARD PREVIOUS RESULT OF 32','DISREGARD PREVIOUSLY REPORTED 33','DISREGARD',"+"], np.nan, inplace=True)
        data.dropna(inplace = True)
        values = data.values
        
        sample = self.extract(values, self.id_name_dict)
        if len(sample[0]) >= self.seq_len and self.uniform != "random":
            sample[0] = sample[0][:self.seq_len]
            sample[1] = sample[1][:self.seq_len]
            sample[2] = sample[2][:self.seq_len]
            sample[3] = sample[3][:self.seq_len]
            
        if self.uniform == "random" and len(sample[0]) >= self.seq_len:
            selected_indices = random.sample(range(len(sample[0])), self.seq_len)

            sample[0] = [sample[0][i] for i in selected_indices]
            sample[1] = [sample[1][i] for i in selected_indices]
            sample[2] = [sample[2][i] for i in selected_indices]
            sample[3] = [sample[3][i] for i in selected_indices]
            
        num_padd_tokens = self.seq_len - len(sample[0])
        variable_input = torch.cat([
            torch.tensor(sample[2], dtype=torch.int64),
            torch.tensor([self.pad_value]*num_padd_tokens, dtype=torch.int64)
        ])
        value_input = torch.cat([
            torch.tensor(sample[1], dtype=torch.float),
            torch.tensor([self.pad_value]*num_padd_tokens, dtype=torch.float)
        ])
        val = torch.tensor(sample[0], dtype=torch.float)
        time_input = torch.cat([
             val - val.min() ,
            torch.tensor([self.pad_value]*num_padd_tokens, dtype=torch.float)
        ])
        variables = sample[3] + ['pad token']*num_padd_tokens
        
        assert variable_input.size(0) == self.seq_len
        assert value_input.size(0) == self.seq_len
        assert time_input.size(0) == self.seq_len
        cols = self.data_df.columns[2:]
        y_true = list(self.data_df.iloc[idx][cols].values)
        return {
            "encoder_input" : [time_input.to(self.device), variable_input.to(self.device), value_input.to(self.device)],
            "encoder_mask": (variable_input != self.pad_value).unsqueeze(0).int().to(self.device),
            "variables" : variables,
            "label" : torch.tensor(y_true, dtype=torch.int64).to(self.device)
        }
    
    def extract(self, values, id_name_dict):
        sample = []
        time = list(values[:, 0])
        variable = list(values[:, 1])
        value = list(values[:, 2])
        count_dict = {}
        
        if self.uniform:
            variable, value, time = self.limit_feature_appearances(variable, value, time, 100)
                
            
        value = [(float(i) - self.mean_variance[var]['mean'])/math.sqrt(self.mean_variance[var]['variance']) for i, var in zip(value, variable)]
        
        varibale_id = [int(id_name_dict[i]) for i in variable]
        sample.append(time)
        sample.append(value)
        sample.append(varibale_id)
        sample.append(variable)
         
        return sample
    

    def limit_feature_appearances(self, feature_names, feature_values, time_values, max_appearances):
        feature_counts = {}

        filtered_data = []
        filtered_feature_names, filtered_feature_values, filtered_time_values = [], [], []
        for i in range(len(feature_names)):
            name = feature_names[i]
            value = feature_values[i]
            time = time_values[i]
            if name not in feature_counts:
                feature_counts[name] = 1
            count = feature_counts[name]
            if count <= max_appearances:
                feature_counts[name] = count + 1
                filtered_feature_names.append(name)
                filtered_feature_values.append(value)
                filtered_time_values.append(time)
        return filtered_feature_names, filtered_feature_values, filtered_time_values
    
    def isNAN(self, val):
        return val!=val

MAX_LEN = 1024
batch_size = 32
d_model = 64
num_heads = 8
N = 2
num_variables = 18 
num_variables += 1 
d_ff = 128
epochs = 75
pretraining = False
learning_rate = 0.0005
drop_out = 0.1
sinusoidal = True
Uniform = "random"
K = 500
import torch
import json

import pandas as pd
import numpy as np
from utils import calculate_roc_auc_bert
pd.set_option('future.no_silent_downcasting',True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F

from model_bert import StratsModel, PredictionModel, Model

from tqdm import tqdm
from normalizer import Normalizer

root_data_dir = "/data/datasets/mimic3_18var/modified_data/data_18_var/"
train_data_path = f"{root_data_dir}phenotyping/train_listfile.csv"
val_data_path = f"{root_data_dir}phenotyping/val_listfile.csv"
test_data_path = f"{root_data_dir}phenotyping/test_listfile.csv"

data_dir = f"{root_data_dir}phenotyping/train/"
test_data_dir = f"{root_data_dir}phenotyping/test/"


import pickle

with open('normalizer.pkl', 'rb') as file:
    normalizer = pickle.load(file)


mean_variance = normalizer.mean_var_dict



strats_model = StratsModel(d_model, num_heads, d_ff, num_variables, N, sinusoidal, pretraining).to(DEVICE)
forecast_model = PredictionModel(d_model).to(DEVICE)
model = Model(strats_model, forecast_model).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

model.load_state_dict(torch.load('models/sinusoidalTrue_Uniformrandom_modellatest_model.pth'))
data = pd.read_csv(test_data_path)

res_dict = {"macro" : [],
           "micro" : []}
for k in range(K):
    resampled_data = data.sample(n = len(data), replace=True)
    resampled_data.reset_index(drop=True, inplace=True)
    test_ds = MimicDataSetPhenotyping(test_data_dir, resampled_data, mean_variance,'testing', MAX_LEN, False)
    test_dataloader = DataLoader(test_ds, batch_size = 64)
    macro, micro, test_loss = calculate_roc_auc_bert(model, test_dataloader)
    res_dict["macro"].append(macro)
    res_dict["micro"].append(micro)

with open("base_line_latest.json","w") as f:
    json.dump(res_dict, f)

    
    
res_dict = {"macro" : [],
           "micro" : []}
for k in range(K):
    resampled_data = data.sample(n = len(data), replace=True)
    resampled_data.reset_index(drop=True, inplace=True)
    test_ds = MimicDataSetPhenotyping(test_data_dir, resampled_data, mean_variance,'testing', MAX_LEN, True)
    test_dataloader = DataLoader(test_ds, batch_size = 64)
    macro, micro, test_loss = calculate_roc_auc_bert(model, test_dataloader)
    res_dict["macro"].append(macro)
    res_dict["micro"].append(micro)
import json
with open("base_line_uniform.json","w") as f:
    json.dump(res_dict, f)

    
    
    
res_dict = {"macro" : [],
           "micro" : []}

for k in range(K):
    resampled_data = data.sample(n = len(data), replace=True)
    resampled_data.reset_index(drop=True, inplace=True)
    test_ds = MimicDataSetPhenotyping(test_data_dir, resampled_data, mean_variance,'testing', MAX_LEN, "random")
    test_dataloader = DataLoader(test_ds, batch_size = 64)
    macro, micro, test_loss = calculate_roc_auc_bert(model, test_dataloader)
    res_dict["macro"].append(macro)
    res_dict["micro"].append(micro)
import json
with open("base_line_random.json","w") as f:
    json.dump(res_dict, f)