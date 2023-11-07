import torch
from torch import nn

class tcn_net(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.tcn1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.tcn2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.sigmoid = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input_data):
        
        hidden = self.tcn2(self.dropout(self.sigmoid(self.tcn1(input_data)))) + input_data  

        return hidden