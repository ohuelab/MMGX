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
# rdkit
from rdkit.Chem import Descriptors
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

    linear_param = dict()
    linear_param['out_lin'] = args.out_channels
    linear_param['num_layers'] = args.num_layers
    linear_param['classification'] = args.class_number
    linear_param['batch_normalize'] = args.batch_normalize
    linear_param['dropout'] = args.dropout

    if args.model == 'Descriptor':
        model = Desc_Model(setting_param, linear_param)
    elif args.model == 'ECFP4':
        model = FP_Model(setting_param, linear_param)
    
    model.double()
    model.apply(reset_weights)

    return model

###################
### Model Class ###
###################

# Combine model
class FP_Model(torch.nn.Module):
    def __init__(self, setting_param, linear_param):
        super(FP_Model, self).__init__()

        self.file = setting_param['file']
        self.schema = setting_param['schema']
        self.reduced = setting_param['reduced']
        self.device = setting_param['device']
        self.dropout = setting_param['dropout']
        self.fp_size = linear_param['out_lin']
        self.last_mol_embedding = None

        # classification
        self.classification = ClassificationLayer(linear_param)

    def forward(self, data):
        
        # atom graph
        smiles = copy.copy(data.smiles)
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.fp_size) for mol in mols]
        x = torch.tensor(fps, dtype=float).type(torch.DoubleTensor).to(self.device)

        # classification layer
        self.last_mol_embedding = x
        x = self.classification(x)
        
        # return
        return x
    
class Desc_Model(torch.nn.Module):
    def __init__(self, setting_param, linear_param):
        super(Desc_Model, self).__init__()

        self.file = setting_param['file']
        self.schema = setting_param['schema']
        self.reduced = setting_param['reduced']
        self.device = setting_param['device']
        self.dropout = setting_param['dropout']
        self.fp_size = linear_param['out_lin']
        self.last_mol_embedding = None

        # prepare features (descriptor)
        self.mol_smiles = generateGraphDataset(self.file).keys()
        datasets_dict = dict()
        for d in self.mol_smiles:
            mol = smiles_to_mol(d, with_atom_index=False)
            datasets_dict[d] = Descriptors.CalcMolDescriptors(mol)
        datasets_df = pd.DataFrame(datasets_dict).transpose()
        datasets_df = datasets_df.fillna(datasets_df.mean())
        self.mol_dict = datasets_df.to_dict(orient='index')

        # descriptor embedding
        self.descriptor = DescriptorLinear(setting_param, linear_param['out_lin'])
        # classification
        self.classification = ClassificationLayer(linear_param)

    def forward(self, data):
        
        # atom graph
        smiles = copy.copy(data.smiles)
        desc = []
        for smile in smiles:
            if smile in self.mol_dict:
                desc.append(list(self.mol_dict[smile].values()))
            else:
                mol = Chem.MolFromSmiles(smile)
                cal = list(Descriptors.CalcMolDescriptors(mol).values())
                cal = [0 if np.isnan(x) else x for x in cal]
                desc.append(cal)
                assert False
        x = torch.tensor(desc, dtype=float).type(torch.DoubleTensor).to(self.device)

        # classification layer
        x = self.descriptor(x)
        self.last_mol_embedding = x
        x = self.classification(x)
        
        # return
        return x
    

# Descriptor extraction
class DescriptorLinear(nn.Module):
    def __init__(self, setting_param, out_lin):
        super(DescriptorLinear, self).__init__()
        self.lin1 = Linear(len(Descriptors._descList), out_lin)
        self.batch_normalize = setting_param['batch_normalize']
        if self.batch_normalize: self.bn = nn.BatchNorm1d(out_lin)

    def forward(self, x):
        x = self.lin1(x)
        if self.batch_normalize and ((x.size(0) != 1 and self.training) or (not self.training)): x = self.bn(x)
        x = F.leaky_relu(x)
        return x


# Classification Layer
class ClassificationLayer(nn.Module):
    def __init__(self, linear_param):
        super(ClassificationLayer, self).__init__()
        out_lin = linear_param['out_lin']
        num_layers = linear_param['num_layers']
        classification = linear_param['classification']
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.layers.append(nn.Linear(out_lin, out_lin//2))
        self.batch_norms.append(nn.BatchNorm1d(num_features=out_lin//2))
        
        for _ in range(num_layers - 2):
            self.layers.append(Linear(out_lin//(2**(_+1)), out_lin//(2**(_+2))))
            self.batch_norms.append(nn.BatchNorm1d(num_features=out_lin//(2**(_+2))))
        
        self.layers.append(Linear(out_lin//(2**(num_layers-1)), classification))
        self.dropout = linear_param['dropout']

    def forward(self, x, fold_number=None):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if (x.size(0) != 1 and self.training) or (not self.training): x = self.batch_norms[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.layers[-1](x)
        return out

