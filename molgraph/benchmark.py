######################
### Import Library ###
######################

# my library
from statistics import harmonic_mean
from molgraph.dataset import *
from molgraph.attentivefp import *
# standard
import numpy as np
import copy as copy
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, GRUCell
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import GATv2Conv, GINEConv, GCNConv
from torch_geometric.nn import global_add_pool

###################
### Model Class ###
###################

# Benchmark model
class GNN_Benchmark(torch.nn.Module):
    def __init__(self, setting_param, feature_graph_param, feature_reduced_param, graph_param, reduced_param, linear_param):
        super(GNN_Benchmark, self).__init__()

        self.file = setting_param['file']
        self.model = setting_param['model']
        self.schema = setting_param['schema']
        self.reduced = setting_param['reduced']
        self.device = setting_param['device']
        self.dropout = setting_param['dropout']

        if self.schema in ['A', 'R_N', 'AR', 'AR_0', 'AR_N']:
            # features
            self.node_feature_graph = NodeLinear(setting_param,
                                                feature_graph_param['node_graph_in_lin'],
                                                feature_graph_param['node_graph_out_lin'])
            self.edge_feature_graph = EdgeLinear(setting_param,
                                                feature_graph_param['edge_graph_in_lin'],
                                                feature_graph_param['edge_graph_out_lin'])
            # graph
            self.GNN_Graph = GNN_Graph(setting_param,
                                    graph_param['in_channels'], 
                                    graph_param['hidden_channels'], 
                                    graph_param['out_channels'], 
                                    graph_param['num_layers'],
                                    graph_param['in_lin'], 
                                    graph_param['out_lin'], 
                                    graph_param['num_layers_self'],
                                    edge_dim=graph_param['edge_dim'],
                                    heads=graph_param['heads'])

        if self.schema in ['R', 'R_0', 'R_N', 'AR', 'AR_0', 'AR_N']:
            self.dict_reducedgraph = dict()
            self.node_feature_reduced = nn.ModuleDict()
            self.edge_feature_reduced = nn.ModuleDict()
            self.GNN_Reduced = nn.ModuleDict()
            
            for g in setting_param['reduced']:
                # dict of reduced
                self.dict_reducedgraph[g] = setting_param['dict_reducedgraph'][g]
                # feature
                self.node_feature_reduced[g] = NodeLinear(setting_param,
                                                        feature_reduced_param[g]['node_reduced_in_lin'],
                                                        feature_reduced_param[g]['node_reduced_out_lin'])
                self.edge_feature_reduced[g] = EdgeLinear(setting_param,
                                                        feature_reduced_param[g]['edge_reduced_in_lin'],
                                                        feature_reduced_param[g]['edge_reduced_out_lin'])
                # reduced
                self.GNN_Reduced[g] = GNN_Reduced(setting_param,
                                                reduced_param[g]['in_channels'], 
                                                reduced_param[g]['hidden_channels'], 
                                                reduced_param[g]['out_channels'], 
                                                reduced_param[g]['num_layers'],
                                                reduced_param[g]['in_lin'], 
                                                reduced_param[g]['out_lin'], 
                                                reduced_param[g]['num_layers_self'],
                                                edge_dim=reduced_param[g]['edge_dim'],
                                                heads=reduced_param[g]['heads'])

        if self.schema in ['R_0', 'AR_0']:
            self.seen_collection = dict()
            for g in setting_param['reduced']:
                self.seen_collection[g] = dict()

        # pooling
        if self.schema in ['R_N']:
        # if self.schema in ['R_N', 'AR_N']:
            self.pool_features = nn.ModuleDict()
            self.pool_layer = dict()
            self.pool_conv = nn.ModuleDict()
            self.pool_bn = nn.ModuleDict()
            self.pool_gru = nn.ModuleDict()
            for g in setting_param['reduced']:
                self.pool_features[g] = Linear(feature_reduced_param[g]['node_reduced_out_lin'], graph_param['out_channels'])
                self.pool_layer[g] = graph_param['num_layers_self']
                self.pool_conv[g] = GATv2Conv(graph_param['out_channels'], graph_param['out_channels'], add_self_loops=False)
                self.pool_bn[g] = nn.BatchNorm1d(graph_param['out_channels'])
                self.pool_gru[g] = GRUCell(graph_param['out_channels'], graph_param['out_channels'])

        # classification
        self.classification = ClassificationLayer(linear_param)
        
        # last molecule embedding
        self.all_mol_embedding = dict()
        self.last_mol_embedding = None

        # attention molecule embedding
        self.all_mol_attention = dict()

        # # explanation
        # self.input_g = None
        # self.final_conv_grads_g = None
        # self.final_conv_acts_g = None
        # self.input_r = None
        # self.final_conv_grads_r = None
        # self.final_conv_acts_r = None

    def generateReduced_pooling(self, g, data, graph, batch, fold_number=0):
        # with torch.no_grad():
        sub_reduced = list()
        # batch_list = torch.bincount(batch).to(self.device) # non-deterministic
        batch_list = np.bincount(batch.cpu()) # deterministic
        graph_list = torch.split(graph, batch_list.tolist())
        for smiles, y, graphlist in zip(data.smiles, data.y, graph_list):
            # check smiles in seen collection?
            if self.schema in ['R_0', 'AR_0'] and smiles in self.seen_collection[g]:
                _d = self.seen_collection[g][smiles]
                d = copy.copy(_d)
                sub_reduced.append(d)
                continue
            # check smiles in dict_reducedgraph from dataset (train/val/test)
            if smiles in self.dict_reducedgraph[g]:
                _d, cliques = self.dict_reducedgraph[g][smiles]
            # if not construct new graph
            else:
                tokenizer = None
                if g == 'substructure':
                    vocab_path = 'vocab/'+self.file+'_'+str(fold_number)+'.txt'
                    tokenizer = Tokenizer(vocab_path)
                _d, cliques = constructReducedGraph(g, smiles, y, tokenizer)
            # check _d is valid
            if _d is not None:
                d = copy.copy(_d)
                x_r_new = torch.Tensor().to(self.device)
                for s, c in zip(d.x_r, cliques):
                    indices = torch.tensor(c).to(self.device)
                    selected = torch.index_select(graphlist, 0, indices)
                    # max
                    # reducedmax = torch.max(selected, 0, True)[0]
                    # s_new = torch.cat((torch.unsqueeze(s.to(self.device), dim=0), reducedmax), -1)
                    # mean
                    # reducedmean = torch.mean(selected, 0, True)
                    # s_new = torch.cat((torch.unsqueeze(s.to(self.device), dim=0), reducedmean), -1)
                    # sum
                    reducedsum = torch.sum(selected, 0, True)
                    s_new = torch.cat((torch.unsqueeze(s.to(self.device), dim=0), reducedsum), -1)
                    x_r_new = torch.cat((x_r_new, s_new), 0)
                    # construct pooling index
                    if self.schema in ['AR_N']:
                        pooling_src = list()
                        pooling_dst = list()
                        for i, c in enumerate(cliques):
                            pooling_src.extend(c)
                            pooling_dst.extend([i]*len(c))
                        pooling_index = torch.tensor(np.array([pooling_src, pooling_dst])).to(self.device)
                        d.pooling_index = pooling_index 
                d.x_r = x_r_new
                sub_reduced.append(d)
                # add processed smiles in seen_collection for R_0, AR_0 because of raw values
                if self.schema in ['R_0', 'AR_0']: 
                    self.seen_collection[g][smiles] = d
            else:
                print('ERROR: Generate Reduced Graph')
                print(_d, cliques, smiles, y)
                assert False
        # constructure dataloader from number of smiles
        loader = DataLoader(sub_reduced, batch_size=len(data.smiles), shuffle=False, follow_batch=['x_r'])
        data_r = next(iter(loader)).to(self.device)
        # if there is no edge in reduced graph
        if data_r.edge_attr_r.shape[0] == 0:
            data_r.edge_index_r = torch.tensor(np.array([[0],[0]])).type(torch.LongTensor).to(self.device)
        return data_r

    def generateReduced_only(self, g, data, fold_number=0):
        # with torch.no_grad():
        sub = list()
        for smiles, y in zip(data.smiles, data.y):
            # check smiles in dict_reducedgraph from dataset (train/val/test)
            if smiles in self.dict_reducedgraph[g]:
                _d, cliques = self.dict_reducedgraph[g][smiles]
            # if not construct new graph
            else:
                tokenizer = None
                if g == 'substructure':
                    vocab_path = 'vocab/'+self.file+'_'+str(fold_number)+'.txt'
                    tokenizer = Tokenizer(vocab_path)
                _d, cliques = constructReducedGraph(g, smiles, y, tokenizer)
            # check _d is valid
            if _d is not None:
                d = copy.copy(_d) 
                # construct pooling index
                if self.schema in ['R_N', 'AR_N']:
                    pooling_src = list()
                    pooling_dst = list()
                    for i, c in enumerate(cliques):
                        pooling_src.extend(c)
                        pooling_dst.extend([i]*len(c))
                    pooling_index = torch.tensor(np.array([pooling_src, pooling_dst])).to(self.device)
                    d.pooling_index = pooling_index 
                sub.append(d)
            else:
                print('ERROR: Generate Reduced Graph')
                print(_d, cliques, smiles, y)
                assert False
        # constructure dataloader from number of smiles
        loader = DataLoader(sub, batch_size=len(data.smiles), shuffle=False, follow_batch=['x_r'])
        data_r = next(iter(loader)).to(self.device)
        # if there is no edge in reduced graph
        if data_r.edge_attr_r.shape[0] == 0:
            data_r.edge_index_r = torch.tensor(np.array([[0],[0]])).type(torch.LongTensor).to(self.device)

        return data_r

    def forward(self, data, return_attention_weights=False, fold_number=0):
        
        # atom graph
        x_g_original = copy.copy(data.x_g)
        if self.schema in ['A', 'R_N', 'AR', 'AR_0', 'AR_N']:
            data.x_g = self.node_feature_graph(data.x_g)
            data.edge_attr_g = self.edge_feature_graph(data.edge_attr_g)
            graph_x, attention_weights_mol_g = self.GNN_Graph(data, data.x_g_batch)
            self.all_mol_embedding['atom'] = graph_x
            self.all_mol_attention['atom'] = attention_weights_mol_g

        # list of reduced graph
        for g in self.reduced:
            # reduced graph information/pooling
            if self.schema in ['R', 'AR']:
                data_r = self.generateReduced_only(g, data, fold_number=fold_number) # without atom layer info
            elif self.schema in ['R_N']:
            # elif self.schema in ['R_N', 'AR_N']:
                data_r = self.generateReduced_only(g, data, fold_number=fold_number) # without atom layer info for pooling
            elif self.schema in ['AR_N']:
                data_r = self.generateReduced_pooling(g, data, self.GNN_Graph.final_conv_acts, data.x_g_batch, fold_number=fold_number)
            elif self.schema in ['R_0', 'AR_0']:
                data_r = self.generateReduced_pooling(g, data, x_g_original, data.x_g_batch, fold_number=fold_number)

            # reduced graph
            if self.schema in ['R', 'R_0', 'AR', 'AR_0', 'AR_N']:
            # if self.schema in ['R', 'R_0', 'AR', 'AR_0']:
                data_r.x_r = self.node_feature_reduced[g](data_r.x_r)
                data_r.edge_attr_r = self.edge_feature_reduced[g](data_r.edge_attr_r)
                reduced_x, attention_weights_mol_r = self.GNN_Reduced[g](data_r, data_r.x_r_batch)
            elif self.schema in ['R_N']:
            # elif self.schema in ['R_N', 'AR_N']:
                data_r.x_r = self.node_feature_reduced[g](data_r.x_r)
                data_r.edge_attr_r = self.edge_feature_reduced[g](data_r.edge_attr_r)
                # pooling
                # out = F.relu(self.pool_features[g](data_r.x_r))
                out = self.pool_features[g](data_r.x_r)
                edge_index = data_r.pooling_index.to(torch.int64)
                for t in range(self.pool_layer[g]):
                    h = self.pool_conv[g]((self.GNN_Graph.final_conv_acts, out), edge_index)
                    if (h.size(0) != 1 and self.training) or (not self.training): h = self.pool_bn[g](h)
                    h = F.elu_(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    out = self.pool_gru[g](h, out)
                    out = F.leaky_relu(out)
                data_r.x_r = out
                reduced_x, attention_weights_mol_r = self.GNN_Reduced[g](data_r, data_r.x_r_batch)
            self.all_mol_embedding[g] = reduced_x
            self.all_mol_attention[g] = attention_weights_mol_r

        # before fully connected layer
        if self.schema in ['AR', 'AR_0', 'AR_N']:
            x = torch.cat([self.all_mol_embedding[emb] for emb in self.all_mol_embedding], dim=1)
        elif self.schema in ['A']:
            x = graph_x
        elif self.schema in ['R', 'R_0', 'R_N']:
            x = torch.cat([self.all_mol_embedding[emb] for emb in self.all_mol_embedding if emb != 'atom'], dim=1)

        # classification layer
        self.last_mol_embedding = x
        x = self.classification(x)
        
        # return
        if return_attention_weights:
            return x, self.all_mol_attention
        else: 
            return x

# Node Embedding
class NodeLinear(nn.Module):
    def __init__(self, setting_param, in_lin, out_lin):
        super(NodeLinear, self).__init__()
        self.lin1 = Linear(in_lin, out_lin)
        self.batch_normalize = setting_param['batch_normalize']
        if self.batch_normalize: self.bn = nn.BatchNorm1d(out_lin)

    def forward(self, x):
        x = self.lin1(x)
        if self.batch_normalize and ((x.size(0) != 1 and self.training) or (not self.training)): x = self.bn(x)
        x = F.leaky_relu(x)
        return x

# Edge Embedding
class EdgeLinear(nn.Module):
    def __init__(self, setting_param, in_lin, out_lin):
        super(EdgeLinear, self).__init__()
        self.lin1 = Linear(in_lin, out_lin)
        self.batch_normalize = setting_param['batch_normalize']
        if self.batch_normalize: self.bn = nn.BatchNorm1d(out_lin)

    def forward(self, x):
        if x.shape[0] != 0:
            x = self.lin1(x)
            if self.batch_normalize and ((x.size(0) != 1 and self.training) or (not self.training)): x = self.bn(x)
            x = F.leaky_relu(x)
            return x
        else:
            return None

# Classification Layer
class ClassificationLayer(nn.Module):
    def __init__(self, linear_param):
        super(ClassificationLayer, self).__init__()
        out_lin = linear_param['out_lin']
        classification = linear_param['classification']
        # self.lin = Linear(out_lin, out_lin)
        # self.bn = nn.BatchNorm1d(num_features=out_lin)
        self.lin1 = Linear(out_lin, out_lin//2)
        self.bn1 = nn.BatchNorm1d(num_features=out_lin//2)
        self.lin2 = Linear(out_lin//2, out_lin//4)
        self.bn2 = nn.BatchNorm1d(num_features=out_lin//4)
        self.lin3 = Linear(out_lin//4, classification)
        # self.lin3 = Linear(out_lin+(out_lin//4), classification)
        self.dropout = linear_param['dropout']

    def forward(self, x):
        # wide
        # h = self.lin(x)
        h = x
        # if h.size(0) != 1: h = self.bn(h)
        # h = F.leaky_relu(h)
        # h = F.dropout(h, p=self.dropout, training=self.training)
        # deep
        x = self.lin1(h)
        if (x.size(0) != 1 and self.training) or (not self.training): x = self.bn1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if (x.size(0) != 1 and self.training) or (not self.training): x = self.bn2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # combine
        # out = torch.cat((h, x), dim=1)
        # out = self.lin3(out)
        out = self.lin3(x)
        # return x
        return out

# Graph Layer
class GNN_Graph(nn.Module):
    def __init__(self, setting_param, in_channels, hidden_channels, out_channels, num_layers, in_lin, out_lin, num_layers_self, edge_dim, heads):
        super(GNN_Graph, self).__init__()
        self.device = setting_param['device']
        self.model = setting_param['model']
        self.dropout = setting_param['dropout']
        self.batch_normalize = setting_param['batch_normalize']
        self.num_layers = num_layers
        self.num_layers_self = num_layers_self

        if 'AttentiveFP' in self.model:
            self.attentivefp = AttentiveFP(in_channels, hidden_channels, out_lin, edge_dim, num_layers, num_layers_self, self.dropout)
        else:
            # GCN
            if self.batch_normalize:
                self.convs, self.bns = GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads)
            else:
                self.convs = GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads)
            # linear
            self.lin = Linear(in_lin, out_lin)
            
        # explanation
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        
    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, data, batch):
        x, edge_index = data.x_g, data.edge_index_g
        edge_attr = data.edge_attr_g if hasattr(data,'edge_attr_g') else None
        edge_index = edge_index.type(torch.int64)

        self.input = x
        attention_weights_mol = None

        if 'AttentiveFP' in self.model:
            out, final_conv_acts, attention_weights_mol = self.attentivefp(x, edge_index, edge_attr, batch)
            self.final_conv_acts = final_conv_acts
        else:
            # Node Embedding:
            for _ in range(self.num_layers):
                if 'GIN' in self.model:
                    h = self.convs[_](x, edge_index, edge_attr=edge_attr)
                else:
                    h = self.convs[_](x, edge_index)
                if self.batch_normalize and h.size(0) != 1: h = self.bns[_](h)
                h = F.elu_(h)
                x = F.dropout(h, p=self.dropout, training=self.training)

            self.final_conv_acts = x
            
            # Molecule Embedding:
            x = global_add_pool(x, batch)

            # Predictor:
            out = F.dropout(x, p=self.dropout, training=self.training)
            out = self.lin(out)

        return out, attention_weights_mol

# Reduced Graph Layer
class GNN_Reduced(torch.nn.Module):
    def __init__(self, setting_param, in_channels, hidden_channels, out_channels, num_layers, in_lin, out_lin, num_layers_self, edge_dim, heads):
        super(GNN_Reduced, self).__init__()
        self.device = setting_param['device']
        self.model = setting_param['model']
        self.dropout = setting_param['dropout']
        self.batch_normalize = setting_param['batch_normalize']
        self.num_layers = num_layers
        self.num_layers_self = num_layers_self
        
        if 'AttentiveFP' in self.model:
            self.attentivefp = AttentiveFP(in_channels, hidden_channels, out_lin, edge_dim, num_layers, num_layers_self, self.dropout)
        else:
            # GCN
            if self.batch_normalize:
                self.convs, self.bns = GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads)
            else:
                self.convs = GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads)    
            # linear
            self.lin = Linear(in_lin, out_lin)
        
        # explanation
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        
    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, data, batch):
        x, edge_index = data.x_r, data.edge_index_r
        edge_attr = data.edge_attr_r if hasattr(data,'edge_attr_r') else None
        edge_index = edge_index.type(torch.int64)

        self.input = x
        attention_weights_mol = None
        
        if 'AttentiveFP' in self.model:
            out, final_conv_acts, attention_weights_mol = self.attentivefp(x, edge_index, edge_attr, batch)
            self.final_conv_acts = final_conv_acts
        else:
            # Node Embedding:
            for _ in range(self.num_layers):
                if 'GIN' in self.model:
                    h = self.convs[_](x, edge_index, edge_attr=edge_attr)
                else:
                    h = self.convs[_](x, edge_index)
                if self.batch_normalize and h.size(0) != 1: h = self.bns[_](h)
                h = F.elu_(h)
                x = F.dropout(h, p=self.dropout, training=self.training)

            self.final_conv_acts = x
            
            # Molecule Embedding:
            x = global_add_pool(x, batch)

            # Predictor:
            out = F.dropout(x, p=self.dropout, training=self.training)
            out = self.lin(out)

        return out, attention_weights_mol


def GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads):
    device = setting_param['device']
    model = setting_param['model']
    batch_normalize = setting_param['batch_normalize']
    num_layers = num_layers
    heads = heads

    convs = nn.ModuleList()
    if batch_normalize: bns = nn.ModuleList()

    if model == 'Benchmark_GCN':
    
        # first layer
        convs.append(GCNConv(in_channels, hidden_channels).to(device))
        if batch_normalize: bns.append(nn.BatchNorm1d(heads * hidden_channels).to(device))
        # hidden layer
        for _ in range(num_layers - 2):
            convs.append(GCNConv(hidden_channels, hidden_channels).to(device))
            if batch_normalize: bns.append(nn.BatchNorm1d(hidden_channels).to(device))
        # last layer
        convs.append(GCNConv(hidden_channels, out_channels).to(device))
        if batch_normalize: bns.append(nn.BatchNorm1d(out_channels).to(device))
    
    elif model == 'Benchmark_GIN':

        # first layer
        lin_gin = nn.Sequential(Linear(in_channels, hidden_channels))
        convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
        if batch_normalize: bns.append(nn.BatchNorm1d(hidden_channels).to(device))
        # hidden layer
        for _ in range(num_layers - 2):
            lin_gin = nn.Sequential(Linear(hidden_channels, hidden_channels))
            convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
            if batch_normalize: bns.append(nn.BatchNorm1d(hidden_channels).to(device))
        # last layer
        lin_gin = nn.Sequential(Linear(hidden_channels, out_channels))
        convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
        if batch_normalize: bns.append(nn.BatchNorm1d(out_channels).to(device))

    if batch_normalize:
        return convs, bns
    else:
        return convs