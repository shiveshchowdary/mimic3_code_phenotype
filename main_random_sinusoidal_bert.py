MAX_LEN = 1024
batch_size = 32
d_model = 64
num_heads = 8
N = 2
num_variables = 18 
num_variables += 1 #for no variable embedding while doing padding
d_ff = 128
epochs = 75
learning_rate = 8e-4
drop_out = 0.2
sinusoidal = True
Uniform = "random"
pretraining = False
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import json

import pandas as pd
pd.set_option('future.no_silent_downcasting',True)
import numpy as np
from utils import calculate_roc_auc_bert


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
from utils import MimicDataSetPhenotyping

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


train_ds = MimicDataSetPhenotyping(data_dir, train_data_path, mean_variance, 'training', MAX_LEN, Uniform)
val_ds = MimicDataSetPhenotyping(data_dir, val_data_path, mean_variance, 'validation', MAX_LEN, Uniform)
test_ds = MimicDataSetPhenotyping(test_data_dir, test_data_path, mean_variance,'testing', MAX_LEN, Uniform)

train_dataloader = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size = batch_size)
test_dataloader = DataLoader(test_ds, batch_size = batch_size)


strats_model = StratsModel(d_model, num_heads, d_ff, num_variables, N, sinusoidal, pretraining).to(DEVICE)
forecast_model = PredictionModel(d_model).to(DEVICE)
model = Model(strats_model, forecast_model).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

best_val_loss = float('inf')
early_stopping_counter = 0
patience = 10 

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
        class_token_mask = torch.ones(mask.size(0), 1, 1).to(mask.device)
        mask = torch.cat((mask, class_token_mask), dim=2)
        outputs = model(inp, mask)
        loss = criterion(outputs, y.float())
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    macro, micro, val_loss = calculate_roc_auc_bert(model, val_dataloader)
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
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print(f"Early stopping after {epoch + 1} epochs.")
        break

ver = "latest"
file_path = f"sinusoidal{sinusoidal}_Uniform{Uniform}_model{ver}"

with open(f"metrics/{file_path}_metrics.json", "w") as f:
    json.dump(metrics, f)
print("Testing...............")

auc_roc, auc_prc, test_loss = calculate_roc_auc_bert(model, test_dataloader)
print(f"Testing AUC-ROC, AUC-PRC, Loss : {auc_roc:.3f}, {auc_prc:.3f}, {test_loss:.3f}")

# Constructing the file path


# Example usage
torch.save(model.state_dict(), "models/"+ f"{file_path}_model.pth")
