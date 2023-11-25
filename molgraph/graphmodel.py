######################
### Import Library ###
######################

# my library
from statistics import harmonic_mean
from molgraph.dataset import *
from molgraph.benchmark import *
from molgraph.gineconv import * # used modified version from torch_geometric
# standard
import numpy as np
import copy as copy
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, GRUCell
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import GATv2Conv # GINEConv (error if edge_attr is None)
from torch_geometric.nn import global_add_pool

######################
### Model Function ###
######################

# reset weight and load model
# Try resetting model weights to avoid weight leakage.
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def load_model(args):
    setting_param = dict()
    setting_param['file'] = args.file
    setting_param['model'] = args.model
    setting_param['schema'] = args.schema
    setting_param['reduced'] = args.reduced
    setting_param['batch_normalize'] = args.batch_normalize
    setting_param['device'] = args.device
    setting_param['dropout'] = args.dropout

    setting_param['dict_reducedgraph'] = dict()
    if setting_param['schema'] in ['R', 'R_0', 'R_N', 'AR', 'AR_0', 'AR_N']:
        for g in setting_param['reduced']:
            setting_param['dict_reducedgraph'][g] = generateReducedGraphDict(args.file, g, vocab_file=args.file+'_0')

    NUM_FEATURES = {'atom': {'node': 79, 'edge': 10},
                    'junctiontree': {'node': 83, 'edge': 6},
                    'cluster': {'node': 83, 'edge': 6},
                    'functional': {'node': 115, 'edge': 20},
                    'pharmacophore': {'node': 6, 'edge': 3},
                    'substructure': {'node': 1024, 'edge': 14}}

    NUM_FEATURES_REDUCED = dict()
    for g in setting_param['reduced']:
        NUM_FEATURES_REDUCED[g] = dict()
        if setting_param['schema'] in ['R', 'AR']:
            NUM_FEATURES_REDUCED[g]['node'] = NUM_FEATURES[g]['node']
            NUM_FEATURES_REDUCED[g]['edge'] = NUM_FEATURES[g]['edge']
        elif setting_param['schema'] in ['R_0', 'AR_0']:
            NUM_FEATURES_REDUCED[g]['node'] = NUM_FEATURES['atom']['node']+NUM_FEATURES[g]['node']
            NUM_FEATURES_REDUCED[g]['edge'] = NUM_FEATURES[g]['edge']
        elif setting_param['schema'] in ['R_N']:
            # NUM_FEATURES_REDUCED[g]['node'] = args.out_channels 
            NUM_FEATURES_REDUCED[g]['node'] = NUM_FEATURES[g]['node']
            NUM_FEATURES_REDUCED[g]['edge'] = NUM_FEATURES[g]['edge']
        elif setting_param['schema'] in ['AR_N']:
            NUM_FEATURES_REDUCED[g]['node'] = args.out_channels+NUM_FEATURES[g]['node']
            # NUM_FEATURES_REDUCED[g]['node'] = NUM_FEATURES[g]['node']
            NUM_FEATURES_REDUCED[g]['edge'] = NUM_FEATURES[g]['edge']

    feature_graph_param = dict()
    feature_graph_param['node_graph_in_lin'] = NUM_FEATURES['atom']['node']
    feature_graph_param['node_graph_out_lin'] = args.in_channels
    feature_graph_param['edge_graph_in_lin'] = NUM_FEATURES['atom']['edge']
    feature_graph_param['edge_graph_out_lin'] = args.edge_dim

    feature_reduced_param = dict()
    for g in setting_param['reduced']:
        feature_reduced_param[g] = dict()
        feature_reduced_param[g]['node_reduced_in_lin'] = NUM_FEATURES_REDUCED[g]['node']
        feature_reduced_param[g]['node_reduced_out_lin'] = args.in_channels
        feature_reduced_param[g]['edge_reduced_in_lin'] = NUM_FEATURES_REDUCED[g]['edge']
        feature_reduced_param[g]['edge_reduced_out_lin'] = args.edge_dim

    graph_param = dict()
    graph_param['in_channels'] = args.in_channels
    graph_param['hidden_channels']= args.hidden_channels  
    graph_param['out_channels'] = args.out_channels 
    graph_param['num_layers'] = args.num_layers
    graph_param['in_lin'] = graph_param['out_channels']
    graph_param['out_lin'] = args.mol_embedding
    graph_param['num_layers_self'] = args.num_layers_self
    graph_param['edge_dim'] = args.edge_dim
    graph_param['heads'] = args.heads
    
    reduced_param = dict()
    for g in setting_param['reduced']:
        reduced_param[g] = dict()
        reduced_param[g]['in_channels'] = args.in_channels
        reduced_param[g]['hidden_channels'] = args.hidden_channels  
        reduced_param[g]['out_channels'] = args.out_channels
        reduced_param[g]['num_layers'] = args.num_layers_reduced # for reduced graph
        reduced_param[g]['in_lin'] = reduced_param[g]['out_channels'] 
        reduced_param[g]['out_lin'] = args.mol_embedding
        reduced_param[g]['num_layers_self'] = args.num_layers_self_reduced # for reduced graph
        reduced_param[g]['edge_dim'] = args.edge_dim
        reduced_param[g]['heads'] = args.heads
    
    linear_param = dict()
    if setting_param['schema'] in ['AR', 'AR_0', 'AR_N']:
        linear_param['out_lin'] = graph_param['out_lin']+sum([reduced_param[g]['out_lin'] for g in reduced_param])
    elif setting_param['schema'] in ['A']:
        linear_param['out_lin'] = graph_param['out_lin']
    elif setting_param['schema'] in ['R', 'R_0', 'R_N']:
        linear_param['out_lin'] = sum([reduced_param[g]['out_lin'] for g in reduced_param])
    
    linear_param['classification'] = args.class_number
    linear_param['batch_normalize'] = args.batch_normalize
    linear_param['dropout'] = args.dropout

    if 'Benchmark' in setting_param['model']:
        # print(setting_param, feature_graph_param, feature_reduced_param, graph_param, reduced_param, linear_param)
        model = GNN_Benchmark(setting_param, feature_graph_param, feature_reduced_param, graph_param, reduced_param, linear_param)
    else:
        # print(setting_param, feature_graph_param, feature_reduced_param, graph_param, reduced_param, linear_param)
        model = GNN_Combine(setting_param, feature_graph_param, feature_reduced_param, graph_param, reduced_param, linear_param)
    
    model.double()
    model.apply(reset_weights)

    return model

###################
### Model Class ###
###################

# Combine model
class GNN_Combine(torch.nn.Module):
    def __init__(self, setting_param, feature_graph_param, feature_reduced_param, graph_param, reduced_param, linear_param):
        super(GNN_Combine, self).__init__()

        self.file = setting_param['file']
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
        with torch.no_grad():
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
        with torch.no_grad():
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

        if self.batch_normalize:
            self.convs, self.bns, self.atom_grus = GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads)
        else:
            self.convs, self.atom_grus = GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads)    
        
        # self.lin_con = Linear(out_channels*num_layers, out_channels)

        # molecule
        self.mol_conv = GATv2Conv(out_channels, in_lin, add_self_loops=False).to(self.device)
        if self.batch_normalize: self.mol_bns = nn.BatchNorm1d(in_lin).to(self.device)
        self.mol_gru = GRUCell(in_lin, out_channels).to(self.device)
        # self.mol_conv = GATv2Conv(out_channels, out_lin, add_self_loops=False).to(self.device)
        # if self.batch_normalize: self.mol_bns = nn.BatchNorm1d(out_lin).to(self.device)
        # self.mol_gru = GRUCell(out_lin, out_lin).to(self.device)
        
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
        att_mol_stack = list()
        
        # Node Embedding:
        # node_embedding = list()
        for _ in range(self.num_layers):
            h = self.convs[_](x, edge_index, edge_attr=edge_attr)
            if self.batch_normalize and ((h.size(0) != 1 and self.training) or (not self.training)): h = self.bns[_](h)
            h = F.elu_(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.atom_grus[_](h, x)
            x = F.leaky_relu(x)
            # node_embedding.append(global_add_pool(x, batch))
        self.final_conv_acts = x
        
        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = F.leaky_relu(global_add_pool(x, batch))
        # Concatenate graph embeddings
        # out = torch.cat(tuple(node_embedding), dim=1)
        # out = F.leaky_relu(self.lin_con(out))

        for t in range(self.num_layers_self):
            h, attention_weights = self.mol_conv((x, out), edge_index, return_attention_weights=True)
            if self.batch_normalize and ((h.size(0) != 1 and self.training) or (not self.training)): h = self.mol_bns(h)
            h = F.elu_(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out)
            out = F.leaky_relu(out)
            att_mol_index, att_mol_weights = attention_weights
            att_mol_stack.append(att_mol_weights)

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin(out)
        # mean of attention weight
        att_mol_mean = torch.mean(torch.stack(att_mol_stack), dim=0)

        return out, (att_mol_index, att_mol_mean)

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
        
        if self.batch_normalize:
            self.convs, self.bns, self.atom_grus = GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads)
        else:
            self.convs, self.atom_grus = GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads)    
        
        # self.lin_con = Linear(out_channels*num_layers, out_channels)

        # molecule
        self.mol_conv = GATv2Conv(out_channels, in_lin, add_self_loops=False).to(self.device)
        if self.batch_normalize: self.mol_bns = nn.BatchNorm1d(in_lin).to(self.device)
        self.mol_gru = GRUCell(in_lin, out_channels).to(self.device)
        # self.mol_conv = GATv2Conv(out_channels, out_lin, add_self_loops=False).to(self.device)
        # if self.batch_normalize: self.mol_bns = nn.BatchNorm1d(out_lin).to(self.device)
        # self.mol_gru = GRUCell(out_lin, out_lin).to(self.device)
        
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
        att_mol_stack = list()
        
        # Node Embedding:
        # node_embedding = list()
        for _ in range(self.num_layers):
            h = self.convs[_](x, edge_index, edge_attr=edge_attr)
            if self.batch_normalize and ((h.size(0) != 1 and self.training) or (not self.training)): h = self.bns[_](h)
            h = F.elu_(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.atom_grus[_](h, x)
            x = F.leaky_relu(x)
            # node_embedding.append(global_add_pool(x, batch))
        self.final_conv_acts = x
        
        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = F.leaky_relu(global_add_pool(x, batch))
        # Concatenate graph embeddings
        # out = torch.cat(tuple(node_embedding), dim=1)
        # out = F.leaky_relu(self.lin_con(out))

        for t in range(self.num_layers_self):
            h, attention_weights = self.mol_conv((x, out), edge_index, return_attention_weights=True)
            if self.batch_normalize and ((h.size(0) != 1 and self.training) or (not self.training)): h = self.mol_bns(h)
            h = F.elu_(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out)
            out = F.leaky_relu(out)
            att_mol_index, att_mol_weights = attention_weights
            att_mol_stack.append(att_mol_weights)

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin(out)
        # mean of attention weight
        att_mol_mean = torch.mean(torch.stack(att_mol_stack), dim=0)
        
        return out, (att_mol_index, att_mol_mean)


def GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads):
    device = setting_param['device']
    model = setting_param['model']
    batch_normalize = setting_param['batch_normalize']
    dropout = setting_param['dropout']
    num_layers = num_layers
    heads = heads

    convs = nn.ModuleList()
    if batch_normalize: bns = nn.ModuleList()
    atom_grus = nn.ModuleList()

    if model == 'GAT':
    
        # first layer
        convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim, add_self_loops=True).to(device))
        if batch_normalize: bns.append(nn.BatchNorm1d(heads * hidden_channels).to(device))
        atom_grus.append(GRUCell(heads * in_channels, hidden_channels).to(device))
        # hidden layer
        for _ in range(num_layers - 2):
            convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=edge_dim, add_self_loops=True).to(device))
            if batch_normalize: bns.append(nn.BatchNorm1d(heads * hidden_channels).to(device))
            atom_grus.append(GRUCell(heads * hidden_channels, hidden_channels).to(device))
        # last layer
        convs.append(GATv2Conv(hidden_channels, out_channels, heads=heads, concat=False, edge_dim=edge_dim, add_self_loops=True).to(device))
        if batch_normalize: bns.append(nn.BatchNorm1d(out_channels).to(device))
        atom_grus.append(GRUCell(hidden_channels, out_channels).to(device))
    
    elif model == 'GIN':

        # latest
        # first layer
        lin_gin = nn.Sequential(GIN_Sequential(in_channels, hidden_channels, dropout))
        convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
        if batch_normalize: bns.append(nn.BatchNorm1d(hidden_channels).to(device))
        atom_grus.append(GRUCell(in_channels, hidden_channels).to(device))
        # hidden layer
        for _ in range(num_layers - 2):
            lin_gin = nn.Sequential(GIN_Sequential(hidden_channels, hidden_channels, dropout))
            convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
            if batch_normalize: bns.append(nn.BatchNorm1d(hidden_channels).to(device))
            atom_grus.append(GRUCell(hidden_channels, hidden_channels).to(device))
        # last layer
        lin_gin = nn.Sequential(GIN_Sequential(hidden_channels, out_channels, dropout))
        convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
        if batch_normalize: bns.append(nn.BatchNorm1d(out_channels).to(device))
        atom_grus.append(GRUCell(hidden_channels, out_channels).to(device))

        # # first layer
        # lin_gin = nn.Sequential(Linear(in_channels, hidden_channels),nn.BatchNorm1d(hidden_channels), nn.LeakyReLU(),
        #                         Linear(hidden_channels, hidden_channels), nn.LeakyReLU())
        # convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
        # if batch_normalize: bns.append(nn.BatchNorm1d(hidden_channels).to(device))
        # atom_grus.append(GRUCell(in_channels, hidden_channels).to(device))
        # # hidden layer
        # for _ in range(num_layers - 2):
        #     lin_gin = nn.Sequential(Linear(hidden_channels, hidden_channels),nn.BatchNorm1d(hidden_channels), nn.LeakyReLU(),
        #                         Linear(hidden_channels, hidden_channels), nn.LeakyReLU())
        #     convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
        #     if batch_normalize: bns.append(nn.BatchNorm1d(hidden_channels).to(device))
        #     atom_grus.append(GRUCell(hidden_channels, hidden_channels).to(device))
        # # last layer
        # lin_gin = nn.Sequential(Linear(hidden_channels, out_channels),nn.BatchNorm1d(out_channels), nn.LeakyReLU(),
        #                         Linear(out_channels, out_channels), nn.LeakyReLU())
        # convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
        # if batch_normalize: bns.append(nn.BatchNorm1d(out_channels).to(device))
        # atom_grus.append(GRUCell(hidden_channels, out_channels).to(device))

        # # first layer
        # lin_gin = nn.Sequential(Linear(in_channels, hidden_channels))
        # convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
        # if batch_normalize: bns.append(nn.BatchNorm1d(hidden_channels).to(device))
        # atom_grus.append(GRUCell(in_channels, hidden_channels).to(device))
        # # hidden layer
        # for _ in range(num_layers - 2):
        #     lin_gin = nn.Sequential(Linear(hidden_channels, hidden_channels))
        #     convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
        #     if batch_normalize: bns.append(nn.BatchNorm1d(hidden_channels).to(device))
        #     atom_grus.append(GRUCell(hidden_channels, hidden_channels).to(device))
        # # last layer
        # lin_gin = nn.Sequential(Linear(hidden_channels, out_channels))
        # convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
        # if batch_normalize: bns.append(nn.BatchNorm1d(out_channels).to(device))
        # atom_grus.append(GRUCell(hidden_channels, out_channels).to(device))

    if batch_normalize:
        return convs, bns, atom_grus
    else:
        return convs, atom_grus


class GIN_Sequential(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super(GIN_Sequential, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.lin1 = Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.leakyrelu1 = nn.LeakyReLU()
        self.lin2 = Linear(hidden_channels, hidden_channels)
        # self.leakyrelu2 = nn.LeakyReLU()
        self.dropout = dropout

    def forward(self, x):
        x = self.lin1(x)
        if (x.size(0) != 1 and self.training) or (not self.training): x = self.bn1(x)
        x = self.leakyrelu1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x