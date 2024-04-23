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

with open("IdNameDict.json","r") as f:
    var_dict = json.load(f)

class MimicDataSetPhenotyping(Dataset):
    def __init__(self, data_dir, csv_file, mean_variance , mode, seq_len, uniform , var_dict = var_dict, pad_value = 0, device = DEVICE):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.seq_len = seq_len
        self.mode = mode
        self.data_df = pd.read_csv(csv_file)
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
        
        if self.uniform == True:
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

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from tqdm import tqdm

criterion = nn.BCEWithLogitsLoss()
def calculate_roc_auc(model, data_loader):
    model.eval()
    all_probabilities = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for inputs in tqdm(data_loader, leave=False):
            outputs = model(inputs['encoder_input'], inputs['encoder_mask'])
            labels = inputs['label']
            logits = torch.sigmoid(outputs)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            all_probabilities.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits_all = np.concatenate(all_probabilities)
    labels_all = np.concatenate(all_labels)
    total_loss = total_loss/len(data_loader)
    
    roc_auc_macro = roc_auc_score(labels_all, logits_all,average = 'macro')
    roc_auc_micro = roc_auc_score(labels_all, logits_all,average = 'micro')
    return roc_auc_macro, roc_auc_micro, total_loss


def calculate_roc_auc_bert(model, data_loader):
    model.eval()
    all_probabilities = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for inputs in tqdm(data_loader, leave=False):
            mask = inputs['encoder_mask']
            class_token_mask = torch.ones(mask.size(0), 1, 1).to(mask.device)
            mask = torch.cat((mask, class_token_mask), dim=2)
            outputs = model(inputs['encoder_input'], mask)
            labels = inputs['label']
            logits = torch.sigmoid(outputs)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            all_probabilities.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits_all = np.concatenate(all_probabilities)
    labels_all = np.concatenate(all_labels)
    total_loss = total_loss/len(data_loader)
    
    roc_auc_macro = roc_auc_score(labels_all, logits_all,average = 'macro')
    roc_auc_micro = roc_auc_score(labels_all, logits_all,average = 'micro')
    return roc_auc_macro, roc_auc_micro, total_loss
