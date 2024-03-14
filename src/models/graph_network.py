import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch.nn import Linear, ReLU
from torch_geometric.nn import HypergraphConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
from torch.nn import Sequential as Seq
from torch.nn import Linear, ReLU
from torch_geometric.nn import GINConv, HypergraphConv
from torch_geometric.nn import global_mean_pool as gavgp, global_max_pool as gmp, global_add_pool as gap




class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)
        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x, return_x=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        if return_x:
            return A, x
        else:
            return A

class GNN_encoder(torch.nn.Module):
    def __init__(self, dim_features, dim_target, layers=[256, 256],pooling='attention',dropout = 0.4,conv='HyperConv',gembed=False,**kwargs):
        """
        Parameters
        ----------
        dim_features : TYPE Int
            DESCRIPTION. Number of features of each node
        dim_target : TYPE Int
            DESCRIPTION. Number of outputs
        layers : TYPE, optional List of number of nodes in each layer
            DESCRIPTION. The default is [6,6].
        pooling : TYPE, optional
            DESCRIPTION. The default is 'max'.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.0.
        conv : TYPE, optional Layer type string {'GINConv','EdgeConv'} supported
            DESCRIPTION. The default is 'GINConv'.
        gembed : TYPE, optional Graph Embedding
            DESCRIPTION. The default is False. Pool node scores or pool node features
        **kwargs : TYPE
            DESCRIPTION.
        Raises
        ------
        NotImplementedError
            DESCRIPTION.
        Returns
        -------
        None.
        """
        super(GNN_encoder, self).__init__()
        self.dropout = dropout
        self.embeddings_dim=layers
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.attn_gate = Attn_Net_Gated(L=dim_target)
        self.pooling = {'max':global_max_pool,'mean':global_mean_pool,'add':global_add_pool, 'attention':GlobalAttention(self.attn_gate)}[pooling]
        self.gembed = gembed #if True then learn graph embedding for final classification (classify pooled node features) otherwise pool node decision scores

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))
                
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.linears.append(Linear(out_emb_dim, dim_target))                
                subnet = Sequential(Linear(2*input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.nns.append(subnet)
                self.convs.append(HypergraphConv(input_emb_dim, input_emb_dim, use_attention=False))
                    
        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        out = 0
        pooling = self.pooling
        Z = 0
        for layer in range(self.no_layers):            
            if layer == 0:
                x = self.first_h(x)
                z = self.linears[layer](x)
                Z+=z
                z_pool, attn_score = pooling(z, batch)
                dout = F.dropout(z_pool, p=self.dropout, training=self.training)
                out += dout
            else:
                x = self.convs[layer-1](x,edge_index)
                if not self.gembed:
                    z = self.linears[layer](x)
                    Z+=z
                    z_pool, attn_score = pooling(z, batch)
                    dout = F.dropout(z_pool, p=self.dropout, training=self.training)
                else:
                    x_pool, attn_score = pooling(x, batch)
                    dout = F.dropout(self.linears[layer](x_pool), p=self.dropout, training=self.training)
                out += dout
        if self.pooling == 'attention':
            return out#, attn_score
        else:
            return out

def pooling_clusters(data, clusters, batch, agg='max'):
    lt_mean_batch = []
    #print('data: {}'.format(torch.isnan(data).any()))
    for b in torch.unique(batch):
        lt_mean_clusters = []
        for c in torch.unique(clusters):
            data_cluster_batch = data[(clusters == c) & (batch == b)]
            if data_cluster_batch.shape[0] == 0:
                mean_cluster = torch.zeros((128)).reshape(1,1,-1)
            else:
            #print('data cluster batch: {}'.format(torch.isnan(data_cluster_batch).any()))
                if agg == 'max':
                    mean_cluster = torch.amax(data_cluster_batch, dim=0).reshape(1,1,-1)
                elif agg == 'mean':
                    mean_cluster = torch.mean(data_cluster_batch, dim=1).reshape(1,1,-1)
                elif agg == 'sum':
                    mean_cluster = torch.sum(data_cluster_batch, dim=1).reshape(1,1,-1)
            lt_mean_clusters.append(mean_cluster)
            torch_mean_cluster = torch.cat(lt_mean_clusters, dim=1)
        lt_mean_batch.append(torch_mean_cluster)
    torch_mean_batch = torch.cat(lt_mean_batch, dim=0)
    return torch_mean_batch


class GNN_clusters(torch.nn.Module):
    def __init__(self, dim_features, dim_target, layers=[256, 256], pooling='mean', dropout=0.2, conv='HyperConv', gembed=False, **kwargs):
        """
        Parameters
        ----------
        dim_features : TYPE Int
            DESCRIPTION. Number of features of each node
        dim_target : TYPE Int
            DESCRIPTION. Number of outputs
        layers : TYPE, optional List of number of nodes in each layer
            DESCRIPTION. The default is [6,6].
        pooling : TYPE, optional
            DESCRIPTION. The default is 'max'.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.0.
        conv : TYPE, optional Layer type string {'GINConv','EdgeConv'} supported
            DESCRIPTION. The default is 'GINConv'.
        gembed : TYPE, optional Graph Embedding
            DESCRIPTION. The default is False. Pool node scores or pool node features
        **kwargs : TYPE
            DESCRIPTION.
        Raises
        ------
        NotImplementedError
            DESCRIPTION.
        Returns
        -------
        None.
        """
        super(GNN_clusters, self).__init__()
        self.dropout = dropout
        self.embeddings_dim = layers
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        # if True then learn graph embedding for final classification (classify pooled node features) otherwise pool node decision scores
        self.gembed = gembed

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(
                    Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))

            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.linears.append(Linear(out_emb_dim, dim_target))
                subnet = Sequential(
                    Linear(2*input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.nns.append(subnet)
                self.convs.append(HypergraphConv(
                    input_emb_dim, input_emb_dim, use_attention=False))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        # has got one more for initial input
        self.linears = torch.nn.ModuleList(self.linears)

    def forward(self, data):

        x, edge_index, batch, clusters = data.x, data.edge_index, data.batch, data.y

        out = 0
        Z = 0
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                z = self.linears[layer](x)
                Z += z
                dout = F.dropout(z, p=self.dropout, training=self.training)
                out += dout
            else:
                x = self.convs[layer-1](x, edge_index)
                if not self.gembed:
                    z = self.linears[layer](x)
                    Z += z
                    dout = F.dropout(z, p=self.dropout, training=self.training)
                else:
                    dout = F.dropout(self.linears[layer](
                        x), p=self.dropout, training=self.training)
                out += dout
        out = pooling_clusters(out, clusters, batch)
        return out