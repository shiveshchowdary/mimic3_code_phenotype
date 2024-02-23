import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
pd.set_option('future.no_silent_downcasting',True)

class MimicDataSetPhenotype(Dataset):
    def __init__(self, data_dir, csv_file, mean_variance , cat_dict, mode, seq_len, pad_value = 0, device = DEVICE):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.seq_len = seq_len
        self.mode = mode
        self.data_df = pd.read_csv(csv_file)
        self.mean_variance = mean_variance
        self.pad_value = pad_value
        self.device = device
        self.cat_dict = cat_dict
    
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
        data.replace(['ERROR','no data','.','-','/','VERIFIED','CLOTTED',"*",'ERROR DISREGARD PREVIOUS RESULT OF 32','DISREGARD PREVIOUSLY REPORTED 33'], np.nan, inplace=True)
        for i in range(len(data.columns)):
            id_name_dict[i] = data.columns[i]
        values = data.values
        sample = self.extract(values, id_name_dict)
        if len(sample[0]) >= self.seq_len :
            sample[0] = sample[0][-self.seq_len:]
            sample[1] = sample[1][-self.seq_len:]
            sample[2] = sample[2][-self.seq_len:]
            sample[3] = sample[3][-self.seq_len:]
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
        return {
            "encoder_input" : [time_input.to(self.device), variable_input.to(self.device), value_input.to(self.device)],
            "encoder_mask": (variable_input != self.pad_value).unsqueeze(0).int().to(self.device),
            "variables" : variables,
            "label" : torch.tensor(self.data_df[cols].values[idx], dtype=torch.int64).to(self.device)
        }
    
    def extract(self, values, id_name_dict):
        sample = [[],[],[],[]]
        for i in range(values.shape[0]):
            time = values[i,0]
            for j in range(1, values.shape[1]):
                if self.isNAN(values[i][j]) == False:
                    if id_name_dict[j] in self.cat_dict.keys():
                        sample[0].append(time)
                        sample[1].append(self.cat_dict[id_name_dict[j]][values[i][j]])
                        sample[2].append(j)
                        sample[3].append(id_name_dict[j])
                    else:
                        mean = self.mean_variance[id_name_dict[j]]['mean']
                        var = self.mean_variance[id_name_dict[j]]['variance']
                        val = (float(values[i][j]) - mean)/var
                        sample[0].append(time)
                        sample[1].append(val)
                        sample[2].append(j)
                        sample[3].append(id_name_dict[j])
        return sample
    def isNAN(self, val):
        return val!=val


from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

def calculate_multi_class_metrics(model, data_loader):
    model.eval()
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for inputs in tqdm(data_loader, leave=False):
            outputs = model(inputs['encoder_input'], inputs['encoder_mask'])
            labels = inputs['label']
            logits = torch.sigmoid(outputs, dim=1)
            all_probabilities.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits_all = np.concatenate(all_probabilities, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)

    # Micro AUC-ROC
    roc_auc_micro = roc_auc_score(labels_all, logits_all, average='micro', multi_class='ovr')

    # Macro AUC-ROC
    roc_auc_macro = roc_auc_score(labels_all, logits_all, average='macro', multi_class='ovr')



    return roc_auc_micro, roc_auc_macro



# def get_mean_var(data, data_dir):
#     categorical_variables = ['Glascow coma scale eye opening', 
#                                  'Glascow coma scale motor response', 
#                                  'Glascow coma scale verbal response']
#     sample_path = data_dir + data['stay'][0]
#     id_name_dict = {}
#     df = pd.read_csv(sample_path)
#     df.drop(labels=categorical_variables, axis=1, inplace=True)
#     for i in range(len(df.columns)):
#         id_name_dict[i] = df.columns[i]
#     variable_values = {k : [] for k in df.columns[1:]}
#     for sample_path in tqdm(data['stay']):
#         sample_path = data_dir+sample_path
#         df = pd.read_csv(sample_path)
#         values = df.values
#         df.drop(labels=categorical_variables, axis=1, inplace=True)
#         cols = df.columns[1:]
#         df = df[cols]
#         values = df.values
#         for i in range(values.shape[0]):
#             for j in range(values.shape[1]):
#                 try :
#                     np.isnan(values[i][j])
#                 except:
#                     print(values[i][j])
#                 if np.isnan(values[i][j]) == False:
#                     variable_values[id_name_dict[j+1]].append(values[i][j])
#     result_dict = {}
#     for feature, values in variable_values.items():
#         mean_value = np.mean(values)
#         variance_value = np.var(values)
#         result_dict[feature] = {'mean': mean_value, 'variance': variance_value}
#     return result_dict


