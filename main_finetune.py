MAX_LEN = 256
batch_size = 32
d_model = 50
num_heads = 4
N = 2
num_variables = 18 
num_variables += 1 #for no variable embedding while doing padding
d_ff = 100
epochs = 75
learning_rate = 5e-4
drop_out = 0.2
sinusoidal = True
Uniform = "random"
pretraining = True
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import json

import pandas as pd
import numpy as np
from utils import calculate_roc_auc
pd.set_option('future.no_silent_downcasting',True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
from utils import MimicDataSetPhenotyping

from model import StratsModel, PredictionModel, Model

from tqdm import tqdm
from normalizer import Normalizer

root_data_dir = "/data/datasets/mimic3_18var/modified_data/data_18_var/"
train_data_path = f"{root_data_dir}phenotyping/train_listfile.csv"
val_data_path = f"{root_data_dir}phenotyping/val_listfile.csv"
test_data_path = f"{root_data_dir}phenotyping/test_listfile.csv"

data_dir = f"{root_data_dir}phenotyping/train/"
test_data_dir = f"{root_data_dir}phenotyping/test/"


import pickle

with open('../code_inhospital_mortality/normalizer.pkl', 'rb') as file:
    normalizer = pickle.load(file)


mean_variance = normalizer.mean_var_dict


train_ds = MimicDataSetPhenotyping(data_dir, train_data_path, mean_variance, 'training', MAX_LEN, Uniform)
val_ds = MimicDataSetPhenotyping(data_dir, val_data_path, mean_variance, 'validation', MAX_LEN, Uniform)
test_ds = MimicDataSetPhenotyping(test_data_dir, test_data_path, mean_variance,'testing', MAX_LEN, Uniform)

train_dataloader = DataLoader(train_ds, batch_size = 32, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size = 32)
test_dataloader = DataLoader(test_ds, batch_size = 32)


strats_model = StratsModel(d_model, num_heads, d_ff, num_variables, N, sinusoidal, pretraining).to(DEVICE)
strats_model.load_state_dict(torch.load('../MaskedLanguageModelling/Models/Pre_Trained_Model_V1.pth'))
forecast_model = PredictionModel(d_model).to(DEVICE)
model = Model(strats_model, forecast_model).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()

learning_rate_pre = 0.0005
learning_rate_pred = 0.0005
optimizer_pre = torch.optim.Adam(model.strats_model.parameters(), lr = learning_rate_pre)
optimizer_pred = torch.optim.Adam(model.out_model.parameters(), lr = learning_rate_pred)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

best_val_loss = float('inf')
best_model_state = None

early_stopping_counter = 0
patience = 7 

metrics = {
    'train_loss' : [],
    'val_loss' : [],
    'macro' : [],
    'micro' : []
}
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
        inp = batch['encoder_input']
        mask = batch['encoder_mask']
        y = batch['label']
        outputs = model(inp, mask)
        loss = criterion(outputs, y.float())
        total_loss += loss.item()
        
        optimizer_pre.zero_grad()
        optimizer_pred.zero_grad()
        loss.backward()
        optimizer_pre.step()
        optimizer_pred.step()
        
    macro, micro, val_loss = calculate_roc_auc(model, val_dataloader)
    metrics['train_loss'].append(total_loss/len(train_dataloader))
    metrics['val_loss'].append(val_loss)
    metrics['macro'].append(macro)
    metrics['micro'].append(micro)
    print(f'Epoch {epoch + 1}/{epochs}, Validation Macro AUC-ROC: {macro:.3f}')
    print(f'Epoch {epoch + 1}/{epochs}, Validation Micro AUC-ROC: {micro:.3f}')
    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss/len(train_dataloader):.3f}')
    print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.3f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print(f"Early stopping after {epoch + 1} epochs.")
        break
    if epoch == 14:
        for param_group in optimizer_pre.param_groups:
            param_group['lr'] /= 5
        for param_group in optimizer_pred.param_groups:
            param_group['lr'] /= 5

ver = "latest_256"
file_path = f"fine_tuned_version{ver}"

with open(f"metrics/{file_path}_metrics.json", "w") as f:
    json.dump(metrics, f)
print("Testing...............")

auc_roc, auc_prc, test_loss = calculate_roc_auc(model, test_dataloader)
print(f"Testing AUC-ROC, AUC-PRC, Loss : {auc_roc:.3f}, {auc_prc:.3f}, {test_loss:.3f}")

# Constructing the file path


# Example usage
torch.save(best_model_state, "models/"+ f"{file_path}_model.pth")
