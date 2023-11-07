import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .GNN_model.model_batch import Net


class DESTGNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self._in_feat       = model_args['num_feat']
        self._hidden_dim    = model_args['num_hidden']
        self._node_dim      = model_args['node_hidden']
        self._output_dim    = model_args['seq_length']
        self._num_nodes     = model_args['num_nodes']
        node_dim            = model_args['node_dim']
        self.device         = model_args['device']
        self.topk           = model_args['topk']
        self.batch_size     = model_args['batch_size']

        self._model_args    = model_args

        # model
        self.forecast_layer = Net(**model_args)

        # tcn embeddings
        self.embed_dim = model_args['embed_dim']
        self.node_dim_tcn = model_args['node_dim_tcn']
        self.temp_dim_tid = model_args['temp_dim_tid']
        self.temp_dim_diw = model_args['temp_dim_diw']

        # time embeddings
        self.time_of_day_size = model_args['time_of_day_size']
        self.day_of_week_size = model_args['day_of_week_size']
        self.time_in_day_emb  = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
        self.day_in_week_emb  = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))

        # node embeddings -- for static_graph 
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

        # node embeddings -- for dynamic_graph 
        self.node_emb   = nn.Parameter(torch.empty(self._num_nodes, self.node_dim_tcn))

        # STEMB_dim
        self.hidden_dim = self.embed_dim + self.node_dim_tcn + self.temp_dim_tid + self.temp_dim_diw 

        # for dyg
        self.emb1 = nn.Embedding(self._num_nodes, node_dim)
        self.emb2 = nn.Embedding(self._num_nodes, node_dim)
        self.idx  = torch.arange(self._num_nodes)

        # data --> embeddings
        self.input_len = model_args['input_len']
        self.time_series_emb_layer = nn.Conv2d(in_channels=self._in_feat * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # dyg_STEMB
        self.dyg_ge_1 = self.create_network()
        self.dyg_ge_2 = self.create_network()

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.time_in_day_emb)
        nn.init.xavier_uniform_(self.day_in_week_emb)


    def create_network(self):
        dropout = 0.1
        return nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(128, 64)),
                ('dropout1', nn.Dropout(dropout)),
                ('sigmoid1', nn.ReLU()),
                ('fc2', nn.Linear(64, 64)),
                ('dropout2', nn.Dropout(dropout)),
                ('sigmoid2', nn.ReLU()),
                ('fc3', nn.Linear(64, 40)),
                ('dropout3', nn.Dropout(dropout))
            ])
        )
    

    def _batch_TCN_graph_constructor(self, **inputs):
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']

        # 1. static_graph: graph only from embedding like MTGNN
        static_graph = F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)
        mask = torch.zeros(self._num_nodes, self._num_nodes).to(static_graph.device)
        mask.fill_(float('0'))
        s1,t1 = static_graph.topk(self.topk,1)
        mask.scatter_(1,t1,s1.fill_(1))
        static_graph = [static_graph*mask]

        # 2. dynamic_graph: graph from data with STMEB
        alpha = 1
        idx = self.idx.to(E_d.device)
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx) # [nodes, node_dim]

        # hidden: data with STMEB
        input_ge = inputs['hidden'].transpose(1,2).squeeze() 
        filter1 = self.dyg_ge_1(input_ge)
        filter2 = self.dyg_ge_2(input_ge)

        nodevec1 = torch.tanh(alpha * torch.mul(nodevec1, filter1))
        nodevec2 = torch.tanh(alpha * torch.mul(nodevec2, filter2))

        # better than nodevec1*nodevec2
        a = torch.matmul(nodevec1, nodevec1.transpose(2, 1))
        adj = F.relu(torch.tanh(alpha * a))

        a,_ = adj.topk(self.topk,dim=2)
        a_min = torch.min(a,dim=-1).values
        a_min = a_min.unsqueeze(-1).repeat(1,1,adj.shape[-1])
        ge = torch.ge(adj, a_min)
        zero = torch.zeros_like(adj)
        dynamic_graph = [torch.where(ge,adj,zero)]

        return static_graph, dynamic_graph


    def _stemb(self, history_data):
        # 1\2 means time & week in data feature 
        t_i_d_data = history_data[..., 1]
        time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]

        d_i_w_data = history_data[..., 2]
        day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]
        
        spatial_emb = self.node_emb.unsqueeze(0).expand(self.batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
        temporal_d_emb = time_in_day_emb.transpose(1, 2).unsqueeze(-1)
        temporal_w_emb = day_in_week_emb.transpose(1, 2).unsqueeze(-1)

        node_emb_u  = self.node_emb_u  # [N, d]
        node_emb_d  = self.node_emb_d  # [N, d]
        history_data = history_data[:, :, :, :self._in_feat]

        return spatial_emb,temporal_d_emb,temporal_w_emb, history_data, node_emb_u, node_emb_d


    def forward(self, history_data):
        """
        Args:
            history_data (Tensor): history data with shape: [B, L, N, C]
        Returns:
            torch.Tensor: prediction data with shape: [B, N, L]  C 表示每个节点的特征数 只预测一个特征所以输出这里省略掉了1
        """

        # 1. Spatial-Temporal Embedding(STEMB) Generator
        spatial_emb,temporal_d_emb,temporal_w_emb, history_data, node_embedding_u, node_embedding_d = self._stemb(history_data) 
        # [B, L, N, num_feat]  [N, d] 

        # 2. history_data 进行TCN卷积 融合 feat*input_len 维度 生成与emb相同的维度以便于concat
        # history_data.shape # [B, L, N, 3]
        history_data_tcn = history_data.transpose(1, 2).contiguous()
        history_data_tcn = history_data_tcn.view(self.batch_size, self._num_nodes, -1).transpose(1, 2).unsqueeze(-1) 
        time_series_emb  = self.time_series_emb_layer(history_data_tcn) 

        # 1. concat STEMB & Data and sent into tcn_net
        hidden = torch.cat([time_series_emb,spatial_emb,temporal_d_emb,temporal_w_emb], dim=1) # hidden [batch,emb*4,nodes,1]

        # STEMB内部验证 ST 的消融实验时 
        # 1. Ablation no spatial_emb
        # hidden = torch.cat([time_series_emb,0*spatial_emb,temporal_d_emb,temporal_w_emb], dim=1) 

        # 2. Ablation no temporal_d_emb
        # hidden = torch.cat([time_series_emb,spatial_emb,0*temporal_d_emb,0*temporal_w_emb], dim=1) 

        # 3. Ablation no spatial_emb & temporal_d_emb
        # hidden = torch.cat([time_series_emb,0*spatial_emb,0*temporal_d_emb,0*temporal_w_emb], dim=1) 

        # 3. Graph Generator
        static_graph, dynamic_graph = self._batch_TCN_graph_constructor(node_embedding_u=node_embedding_u, node_embedding_d=node_embedding_d,history_data=history_data,hidden=hidden)
        # [N, N]  [batch, N, N]

        # 4. GCN+TCN model 时空网络模型 
        forecast_hidden = self.forecast_layer(time_series_emb, dynamic_graph, static_graph, spatial_emb,temporal_d_emb,temporal_w_emb)  

        forecast = forecast_hidden.transpose(1,2).squeeze()
        
        return forecast # [batch, N, sequence] 