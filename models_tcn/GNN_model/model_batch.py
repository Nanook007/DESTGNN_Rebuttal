import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import *

from .TCN import tcn_net

class Net(nn.Module):
    def __init__(self, propalpha=0.05, **model_args):
        super(Net,self).__init__()

        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim_tcn"]
        self.embed_dim = model_args["embed_dim"]

        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        # MODEL layers
        self.layers = model_args["gcn_layers"]
        # TCN layers
        self.num_layer = model_args["num_layer"]

        skip_channels = model_args["skip_channels"]
        gcn_depth =  model_args["gcn_depth"]
        seq_length =  model_args["seq_length"]
        dropout =  model_args["dropout"]

        self.skip_convs = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.encoder_fc = nn.ModuleList()
        self.gconv  =  nn.ModuleList()
        self.gconv1 =  nn.ModuleList()
        self.gconv2 =  nn.ModuleList()

        # STEMB_dim
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_day_in_week) + \
            self.temp_dim_diw*int(self.if_time_in_day)
        # self.hidden_dim = self.embed_dim

        for _ in range(self.layers):
            self.skip_convs.append(nn.Conv2d(in_channels=self.hidden_dim, out_channels=skip_channels, kernel_size=(1, 1)))
            self.norm.append(LayerNorm((self.embed_dim, self.num_nodes, 1),elementwise_affine=True))

            # stacked tcn_net
            self.encoder = nn.Sequential(*[tcn_net(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
            self.encoder_fc.append(self.encoder)

            self.gconv.append(mixprop(self.embed_dim, self.embed_dim, gcn_depth, dropout, propalpha))
            self.gconv1.append(mixprop_ge_batch(self.embed_dim, self.embed_dim, gcn_depth, dropout, propalpha))
            self.gconv2.append(mixprop_ge_batch(self.embed_dim, self.embed_dim, gcn_depth, dropout, propalpha))

        
        device = model_args['device']
        self.idx = torch.arange(self.num_nodes).to(device)
        self.skipE = nn.Conv2d(in_channels=self.embed_dim, out_channels=skip_channels, kernel_size=(1, 1), bias=True)
        self.end_conv = nn.Conv2d(in_channels=skip_channels, out_channels=seq_length, kernel_size=(1,1), bias=True)

    def forward(self, input, dy_graph, static_graph, spatial_emb,temporal_d_emb,temporal_w_emb):
        
        # input [batch,emb,nodes,1]
        x = input
        adp = dy_graph[0]

        skip = 0
        for i in range(self.layers):

            residual = x
            time_series_emb = x

            # 1. concat STEMB & Data and sent into tcn_net
            hidden = torch.cat([time_series_emb,spatial_emb,temporal_d_emb,temporal_w_emb], dim=1) # hidden [batch,emb*4,nodes,1]

            # STEMB内部验证 ST 的消融实验时
            # 1. Ablation no spatial_emb
            # hidden = torch.cat([time_series_emb,0*spatial_emb,temporal_d_emb,temporal_w_emb], dim=1) 

            # 2. Ablation no temporal_d_emb
            # hidden = torch.cat([time_series_emb,spatial_emb,0*temporal_d_emb,0*temporal_w_emb], dim=1) 

            # 3. Ablation no spatial_emb & temporal_d_emb
            # hidden = torch.cat([time_series_emb,0*spatial_emb,0*temporal_d_emb,0*temporal_w_emb], dim=1) 

            hidden = self.encoder_fc[i](hidden) # hidden [batch,emb*4,nodes,1]

            # 2. sum output of layers
            s = self.skip_convs[i](hidden) # hidden [batch,skip_channels,nodes,1]
            skip = s + skip

            # 3. Update the input to the next layer through GNN
            x = self.gconv[i](x, static_graph[0]) + self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(-1,-2))
            x = x + residual
            x = F.relu(self.norm[i](x,self.idx))  # [batch, embed_dim, nodes, 1])

        # 4. sum output of layers and the final updated data
        # skip_channels --> seq_length
        skip = self.skipE(x) + skip
        x = self.end_conv(skip) # [batch, seq_length, nodes, 1])

        return x 