import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from utils import format_esm

class LayerNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1_1 = nn.Linear(2560, hidden_dim, dtype=dtype, device=device)
        self.fc1_2 = nn.Linear(2048, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

        self.pred_fc = nn.Sequential(
            nn.Linear(out_dim, 5242, dtype=dtype, device=device),
            nn.LayerNorm(5242, dtype=dtype, device=device),
            nn.Dropout(p=drop_out),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        esm_x, con_x = x[:, :2560], x[:, 2560:]
        x = self.dropout(self.ln1(self.fc1_1(esm_x) + self.fc1_2(con_x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        
        prediction = self.pred_fc(x) 
        return x,prediction


class LayerNormNet_1(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(LayerNormNet_1, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype
        
        self.flatten = nn.Flatten()
        
        self.conv_1 = nn.Sequential(
        nn.Conv1d(1, 2, kernel_size=3, stride=1, padding=0, dtype=dtype, device=device),
        nn.BatchNorm1d(2, dtype=dtype, device=device),
        nn.ReLU(),
        nn.MaxPool1d(3, stride = 3))       

        out1_size = self.compute_conv_output_size(512, 3, 1, 0)
        maxpool1_size = self.compute_pool_output_size(out1_size, 3, 3)
        
        self.fc1_1 = nn.Linear(2048, hidden_dim, dtype=dtype, device=device)
        self.fc1_2 = nn.Linear(2560, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(maxpool1_size*2, hidden_dim,
                             dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        esm_x, con_x = x[:, :2560], x[:, 2560:]
        x = self.dropout(self.ln1(self.fc1_1(con_x) + self.fc1_2(esm_x)))
        x = torch.relu(x)
        
        res = x
        x=self.flatten(self.conv_1( torch.unsqueeze(x, dim=1)))
        
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = x+res
        x = self.fc3(x)
        return x
    def compute_conv_output_size(self, input_size, kernel_size, stride, padding):
            return int((input_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)

    def compute_pool_output_size(self, input_size, pool_size, pool_stride):
        return int((input_size - (pool_size - 1) - 1) / pool_stride + 1)



class ProteinDataset(Dataset):
    def __init__(self, dataframe, esm_dir):
        self.data = dataframe
        self.esm_dir = esm_dir
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data.iloc[idx]['Entry']
        esm_features = torch.load(os.path.join(self.esm_dir, f"{entry}.pt"))
        esm_features = format_esm(esm_features)  
        return {
            'features': esm_features.float(),
        }


class EnzNonEnzBinary(nn.Module):
    def __init__(self, hidden_dim=512, out_dim=128, drop_out=0.3):
        super().__init__()
        self.fc_esm = nn.Linear(2560, hidden_dim)
        self.fc_con = nn.Linear(2048, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        
        self.pred_fc = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Dropout(drop_out),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(drop_out)
        
    def forward(self, x):
        esm_x, con_x = x[:, :2560], x[:, 2560:]
        x = self.fc_esm(esm_x) + self.fc_con(con_x)
        x = F.relu(self.ln1(self.dropout(x)))
        x = F.relu(self.ln2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = self.pred_fc(x)
        return x.squeeze(-1)

