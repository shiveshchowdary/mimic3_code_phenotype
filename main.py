MAX_LEN = 448
batch_size = 32
d_model = 64
num_heads = 8
N = 2
num_variables = 18 
num_variables += 1 #for no variable embedding while doing padding
d_ff = 128
epochs = 50
learning_rate = 8e-4
drop_out = 0.2
sinusoidal = True
th_val_roc = 0.84
th_val_pr = 0.48
num_classes = 25
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import pandas as pd
import numpy as np
from utils import MimicDataSetPhenotype, calculate_multi_class_metrics
pd.set_option('future.no_silent_downcasting',True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F

from model import Model
from tqdm import tqdm
from normalizer import Normalizer
from categorizer import Categorizer


train_data_path = "/data/datasets/mimic3_18var/root/phenotyping/train_listfile.csv"
val_data_path = "/data/datasets/mimic3_18var/root/phenotyping/val_listfile.csv"

data_dir = "/data/datasets/mimic3_18var/root/phenotyping/train/"


import pickle

with open('normalizer.pkl', 'rb') as file:
    normalizer = pickle.load(file)

with open('categorizer.pkl', 'rb') as file:
    categorizer = pickle.load(file)
    

mean_variance = normalizer.mean_var_dict
cat_dict = categorizer.category_dict


train_ds = MimicDataSetPhenotype(data_dir, train_data_path, mean_variance, cat_dict, 'training', MAX_LEN)
val_ds = MimicDataSetPhenotype(data_dir, val_data_path, mean_variance, cat_dict, 'validation', MAX_LEN)
# test_ds = MimicDataSetPhenotype(test_data_dir, test_data_path, mean_variance, cat_dict,'testing', MAX_LEN)

train_dataloader = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=True)
# test_dataloader = DataLoader(test_ds, batch_size = 1, shuffle=True)

model = Model(d_model, num_heads, d_ff, num_classes, N, sinusoidal).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

for epoch in range(epochs):
    # roc_auc_micro, roc_auc_macro = calculate_multi_class_metrics(model, val_dataloader)
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
        inp = batch['encoder_input']
        mask = batch['encoder_mask']
        y = batch['label']
        outputs = model(inp, mask)
        # print(outputs.shape)
        # print(y.shape)
        loss = criterion(outputs.view(-1), y.float().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Loss", loss.item())
    # roc_auc_micro, roc_auc_macro = calculate_multi_class_metrics(model, val_dataloader)
    # # print(f'Epoch {epoch + 1}/{epochs}, Train AUC-ROC: {calculate_roc_auc(model, train_dataloader):.3f}')
    # print(f'Epoch {epoch + 1}/{epochs}, Validation Micro AUC-ROC: {roc_auc_micro:.3f}')
    # print(f'Epoch {epoch + 1}/{epochs}, Validation Macro AUC-ROC: {roc_auc_macro:.3f}')
    # if (auc_prc > th_val_pr) or (auc_roc > th_val_roc):
    #     print("Reached threshold limit stopping...............")
    #     break

# print("Testing...............")
# print(f"Validation AUC-ROC, AUC-PRC: {calculate_roc_auc(model, test_dataloader):.3f}, {calculate_auc_prc(model, test_dataloader):.3f}")

# Constructing the file path
file_path = f"model_maxlen{MAX_LEN}_batch{batch_size}_dmodel{d_model}_heads{num_heads}_N{N}_vars{num_variables}_dff{d_ff}_epochs{epochs}_lr{learning_rate}_dropout{drop_out}_sinusoidal{sinusoidal}_testing.pth"

# Example usage
torch.save(model.state_dict(), "models/"+ file_path)
