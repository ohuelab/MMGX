######################
### Import Library ###
######################

# my library
from molgraph.molgraph import *
from molgraph.substructurevocab import *
from molgraph.utilsmol import *
# standard
import os as os
import pandas as pd
import numpy as np
import pickle as pickle
import csv as csv
from tqdm import tqdm
# pytorch
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# deepchem
import deepchem as dc

########################
### Dataset Function ###
########################

# write disk dataset to CSV file
def writeToCSV(datasets, path):
    datasets.to_csv(path, index=False)

# write disk dataset to CSV file
def readFromCSV(path):
    datasets = pd.read_csv(path)
    return datasets

# Dataset getting and construction dataframe
def getDatasetFromFile(file, smiles, task, splitting):
    path = 'dataset/'+file+'.csv'

    if os.path.exists(path):
        df = pd.read_csv('dataset/'+file+'.csv')
        df_new = df[[smiles, task]].copy()
        df_new.columns = ['X', 'y']
        df_new['ids'] = list(df[smiles]) # original smiles
        if splitting not in ['random', 'scaffold']:
            df_new['s'] = df[splitting]
        else:
            df_new['s'] = np.zeros(len(df[smiles]))
        datasets = df_new
        print('Function: getDatasetFromFile()')
        print("number of all smiles:", len(list(df_new['X'])))
    else:
        print('ERROR: file does not exist.')

    return datasets

# Recheck all valid smiles in dataset
def getValidDataset(datasets):
    not_sucessful = 0
    single_atom = 0
    processed_smiles = 0
    X = datasets['X']
    
    valid_smiles = list()
    validated = list()
    for smiles in tqdm(X.values, desc='===Checking valid smiles==='):
        try:
            smiles_valid = getValidSmiles(smiles)
            # not included single atom
            if checkSingleAtom(smiles_valid):
                single_atom += 1
                valid_smiles.append(smiles)
                validated.append('invalid')
                continue
            processed_smiles += int(smiles_valid != smiles)
            valid_smiles.append(smiles_valid)
            validated.append('valid')
        except:
            not_sucessful += 1
            valid_smiles.append(smiles)
            validated.append('invalid')
            pass
    
    datasets['X'] = valid_smiles
    datasets['v'] = validated
    print('Function: getValidDataset()')
    print("number of all smiles:", len(list(X)))
    print("number of valid smiles:", validated.count('valid'))
    print("number of failed smiles (rdkit):", not_sucessful)
    print("number of failed smiles (single atom):", single_atom)
    print("number of processed smiles:", processed_smiles)

    new_datasets = datasets
    return new_datasets

# remove conflict smiles (Same X diff y)
def getNonConflictDataset(datasets):
    result = datasets.groupby('X')['y'].transform('nunique') > 1
    dataset_conflict = datasets[result]
    print("number of conflict smiles:", len(list(dataset_conflict['X'])))
    new_datasets = datasets
    new_datasets.loc[result, 'v'] = 'invalid'
    print("number of valid smiles:", list(datasets['v']).count('valid'))
    return new_datasets

# get validated dataset 
def getDataset(file, smiles, task, splitting):
    # creating file folder
    if not(os.path.isdir('./dataset/{}'.format(file))):
        os.makedirs(os.path.join('./dataset/{}'.format(file)))
    # read validated if exists
    path = 'dataset/'+file+'/'+file+'_validated.csv'
    if os.path.exists(path):
        datasets = readFromCSV(path)
        datasets = datasets[datasets['v']=='valid']
    else:
        # retrieve data from file
        datasets = getDatasetFromFile(file, smiles, task, splitting)
        # validate smiles
        datasets = getValidDataset(datasets)
        # remove conflict smiles (Same X diff y)
        datasets = getNonConflictDataset(datasets)
        # write to files
        writeToCSV(datasets, path)  

    datasets = datasets[datasets['v']=='valid']
    print('Function: getDataset()')
    print("number of valid smiles:", len(list(datasets['X'])))
    return datasets

def getPosWeight(datasets):
    return np.sum((datasets['y']==0))/np.sum((datasets['y']==1))


# generate dataset with splitting
def generateDatasetSplitting(file, splitting=None, splitting_fold=None, splitting_seed=None):
    path = 'dataset/'+file+'/train_'+str(splitting_seed)+'_'+str(splitting_fold-1)+'.csv'
    
    if os.path.exists(path):
        datasets_fold = []
        for i in range(splitting_fold):
            path_train = 'dataset/'+file+'/train_'+str(splitting_seed)+'_'+str(i)+'.csv'
            path_val = 'dataset/'+file+'/val_'+str(splitting_seed)+'_'+str(i)+'.csv'
            datasets_fold.append((readFromCSV(path_train), readFromCSV(path_val)))
        path_test = 'dataset/'+file+'/test.csv'
        datasets_test = readFromCSV(path_test)
    else:
        # read validated if exists
        path = 'dataset/'+file+'/'+file+'_validated.csv'
        if os.path.exists(path):
            datasets = readFromCSV(path)
            datasets = datasets[datasets['v']=='valid']
            # datasets = datasets.sample(frac=1) # in case shuffle is needed
        
        # construct splitting method
        if splitting == 'random':
            print(splitting, 'RandomSplitter')
            # convert to DiskDataset using smiles as ids (original smiles)
            datasets = dc.data.DiskDataset.from_numpy(X=datasets['X'].values, y=datasets['y'].values, ids=datasets['ids'].values)
            splitter_s = dc.splits.RandomSplitter()
            # return Numpy Dataset object x, y, w, ids
            if splitting_fold != 1:
                datasets_trainval, datasets_test = splitter_s.train_test_split(dataset=datasets, seed=splitting_seed)
                datasets_fold = splitter_s.k_fold_split(dataset=datasets_trainval, k=splitting_fold)
            else: # no cross validation
                datasets_train, datasets_val, datasets_test = splitter_s.train_valid_test_split(dataset=datasets, seed=splitting_seed)
                datasets_fold = [(datasets_train, datasets_val)]

        elif splitting == 'scaffold':
            print(splitting, 'ScaffoldSplitter')
            # convert to DiskDataset using smiles as ids (original smiles)
            datasets = dc.data.DiskDataset.from_numpy(X=datasets['X'].values, y=datasets['y'].values, ids=datasets['ids'].values)
            splitter_s = dc.splits.ScaffoldSplitter()
            # return Numpy Dataset object x, y, w, ids
            if splitting_fold != 1:
                datasets_trainval, datasets_test = splitter_s.train_test_split(dataset=datasets, seed=splitting_seed)
                datasets_fold = splitter_s.k_fold_split(dataset=datasets_trainval, k=splitting_fold)
            else: # no cross validation
                datasets_train, datasets_val, datasets_test = splitter_s.train_valid_test_split(dataset=datasets, seed=splitting_seed)
                datasets_fold = [(datasets_train, datasets_val)]

        else:
            print(splitting, 'Defined')
            datasets_trainval_X = list()
            datasets_trainval_y = list()
            datasets_trainval_ids = list()
            datasets_test_X = list()
            datasets_test_y = list()
            datasets_test_ids = list()
            for idx, row in datasets.iterrows():
                if row['s'].lower() != 'test':
                    datasets_trainval_X.append(row['X'])
                    datasets_trainval_y.append(row['y'])
                    datasets_trainval_ids.append(row['ids'])
                elif row['s'].lower() == 'test':
                    datasets_test_X.append(row['X'])
                    datasets_test_y.append(row['y'])
                    datasets_test_ids.append(row['ids'])

            datasets_trainval = dc.data.DiskDataset.from_numpy(X=datasets_trainval_X, y=datasets_trainval_y, ids=datasets_trainval_ids)
            datasets_test = dc.data.DiskDataset.from_numpy(X=datasets_test_X, y=datasets_test_y, ids=datasets_test_ids)
            # return Numpy Dataset object x, y, w, ids
            splitter_s = dc.splits.RandomSplitter()
            if splitting_fold != 1:
                datasets_fold = splitter_s.k_fold_split(dataset=datasets_trainval, k=splitting_fold)
            else:
                datasets_train, datasets_val = splitter_s.train_test_split(dataset=datasets_trainval, frac_train=0.9, seed=splitting_seed)
                datasets_fold = [(datasets_train, datasets_val)]
                
        datasets_fold_df = list()
        # write dataset to csv
        for i in range(len(datasets_fold)):
            train_df = datasets_fold[i][0].to_dataframe()
            val_df = datasets_fold[i][1].to_dataframe()
            writeToCSV(train_df, 'dataset/'+file+'/train_'+str(splitting_seed)+'_'+str(i)+'.csv') 
            writeToCSV(val_df, 'dataset/'+file+'/val_'+str(splitting_seed)+'_'+str(i)+'.csv')  
            datasets_fold_df.append((train_df, val_df))
        test_df = datasets_test.to_dataframe()
        writeToCSV(test_df, 'dataset/'+file+'/test.csv')
        datasets_fold = datasets_fold_df
        datasets_test = test_df

    datasets_splitted = (datasets_fold, datasets_test) 
    print('Function: generateDatasetSplitting()')
    print('Fold:', len(datasets_fold))
    for i, f in enumerate(datasets_fold):
        print('Fold Number:', i)
        print('-- Datasets Train:', len(list(f[0]['X'])))
        print('-- Datasets Val:', len(list(f[1]['X'])))
        print('-- Datasets Test: ', len(datasets_test['X']))
        print('-- Total:', len(list(f[0]['X']))+len(list(f[1]['X']))+len(datasets_test['X']))

    return datasets_splitted

##################
### Data Class ###
##################

# Pair data class
class PairData(Data):
    def __init__(self, edge_index_g=None, x_g=None, edge_index_r=None, x_r=None, **kwargs):
        super().__init__()
        
        if x_g is not None:
            self.x_g = x_g
        if edge_index_g is not None:
            self.edge_index_g = edge_index_g
        if x_r is not None:
            self.x_r = x_r
        if edge_index_r is not None:
            self.edge_index_r = edge_index_r
            
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_g':
            return self.x_g.size(0)
        elif key == 'edge_index_r':
            return self.x_r.size(0)
        elif key == 'pooling_index_g':
            return self.num_nodes_g.item()
        elif key == 'pooling_index_r':
            return self.num_nodes_r.item()
        else:
            return super().__inc__(key, value, *args, **kwargs)


# Generate Pair data for atom-graph
def constructGraph(smiles, y, ids=None):
    try:
        # pair data
        d = PairData()
        d.smiles = smiles
        if ids: d.ids = ids
        atom_graph = AtomGraph(smiles)
        graph_size, node_index, node_features, edge_index, edge_features = atom_graph.graph_size, atom_graph.node_index, atom_graph.node_features, atom_graph.edge_index, atom_graph.edge_features

        # feature values
        d.x_g = torch.tensor(np.array(node_features, dtype=float)).type(torch.DoubleTensor)
        if graph_size == 1:
            print('WARNING (SINGLE ATOM):', smiles)
            edge_index = [[],[]]
            d.edge_index_g = torch.tensor(np.array(edge_index))
            d.edge_attr_g = torch.Tensor()
        else:
            if len(edge_index) == 0:
                d.edge_index_g = torch.tensor(np.array([[],[]])).type(torch.LongTensor)
            else:
                d.edge_index_g = torch.transpose(torch.tensor(edge_index), 0, 1)
            if len(edge_features) == 0:
                d.edge_attr_g = torch.Tensor(np.array([]))
            else:
                d.edge_attr_g = torch.tensor(np.array(edge_features, dtype=float)).type(torch.DoubleTensor)
        # predicting value
        d.y = torch.tensor([[y]]).type(torch.DoubleTensor)
        return d

    except Exception as e:
        print('ERROR (MOL FAIL):', smiles, e)
        return None

# Generate Pair data for reduced-graph
def constructReducedGraph(reducedgraph, smiles, y, tokenizer=None):
    try:
        # pair data
        d = PairData()
        d.smiles = smiles
    
        # reduced graph
        if reducedgraph == 'atom':
            atom_graph = AtomGraph(smiles)
            cliques = atom_graph.cliques
            graph_size, node_index, node_features, edge_index, edge_features = atom_graph.graph_size, atom_graph.node_index, atom_graph.node_features, atom_graph.edge_index, atom_graph.edge_features
        elif reducedgraph == 'junctiontree':
            junctiontree_graph = JunctionTreeGraph(smiles)
            cliques = junctiontree_graph.cliques
            graph_size, node_index, node_features, edge_index, edge_features = junctiontree_graph.graph_size, junctiontree_graph.node_index, junctiontree_graph.node_features, junctiontree_graph.edge_index, junctiontree_graph.edge_features
        elif reducedgraph == 'cluster':
            cluster_graph = ClusterGraph(smiles)
            cliques = cluster_graph.cliques
            graph_size, node_index, node_features, edge_index, edge_features = cluster_graph.graph_size, cluster_graph.node_index, cluster_graph.node_features, cluster_graph.edge_index, cluster_graph.edge_features
        elif reducedgraph == 'functional':
            functional_graph = FunctionalGraph(smiles)
            cliques = functional_graph.cliques
            graph_size, node_index, node_features, edge_index, edge_features = functional_graph.graph_size, functional_graph.node_index, functional_graph.node_features, functional_graph.edge_index, functional_graph.edge_features
        elif reducedgraph == 'pharmacophore':
            pharmacophore_graph = PharmacophoreGraph(smiles)
            cliques = pharmacophore_graph.cliques
            graph_size, node_index, node_features, edge_index, edge_features = pharmacophore_graph.graph_size, pharmacophore_graph.node_index, pharmacophore_graph.node_features, pharmacophore_graph.edge_index, pharmacophore_graph.edge_features
        elif reducedgraph == 'substructure':
            substructure_graph = SubstructureGraph(smiles, tokenizer)
            cliques = substructure_graph.cliques
            graph_size, node_index, node_features, edge_index, edge_features = substructure_graph.graph_size, substructure_graph.node_index, substructure_graph.node_features, substructure_graph.edge_index, substructure_graph.edge_features

        # feature values
        d.x_r = torch.tensor(np.array(node_features, dtype=float)).type(torch.DoubleTensor)
        if graph_size > 1:
            if len(edge_index) == 0:
                d.edge_index_r = torch.tensor(np.array([[],[]])).type(torch.LongTensor)
            else:
                d.edge_index_r = torch.transpose(torch.tensor(edge_index), 0, 1)
            if len(edge_features) == 0:
                d.edge_attr_r = torch.Tensor(np.array([]))
            else:
                d.edge_attr_r = torch.tensor(np.array(edge_features, dtype=float)).type(torch.DoubleTensor) 
        else:
            d.edge_index_r = torch.tensor(np.array([[],[]])).type(torch.LongTensor)
            d.edge_attr_r = torch.Tensor(np.array([]))
        # predicting value
        d.y = torch.tensor([[y]]).type(torch.DoubleTensor)
        return d, cliques

    except Exception as e:
        print('ERROR (REDUCED FAIL):', smiles, e)
        return None, None

# generate graph fron (xi, yi, ids)
def generateGraph(d, with_id=False):
    smiles, y, ids = d
    if with_id:
        g = constructGraph(smiles, y, ids)
    else:
        g = constructGraph(smiles, y)
    return g

# generate graph datasets
def generateGraphDataset(file, datasets=None):
    path = 'dataset/'+file+'/graph.pickle'

    if os.path.exists(path):
        with open(path, 'rb') as handle:
            datasets_graph = pickle.load(handle)
    else:
        # read validated if exists
        path_validate = 'dataset/'+file+'/'+file+'_validated.csv'
        if os.path.exists(path_validate):
            datasets = readFromCSV(path_validate)
            datasets = datasets[datasets['v']=='valid']

        datasets_graph = dict()
        for d in tqdm(zip(datasets['X'].values, datasets['y'].values, datasets['ids'].values)):
            smiles = d[0]
            g = generateGraph(d)
            if g:
                datasets_graph[smiles] = g

        # write dataset to pickle 
        with open(path, 'wb') as handle:
            pickle.dump(datasets_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        print('Function: generateGraphDataset()')
        print('Datasets graph: ', len(datasets_graph))
    except:
        print('Cannot print dataset statistics')

    return datasets_graph

# generate graph datasets
def generateGraphDatasetUnknown(file, smiles):
    # creating file folder
    if not(os.path.isdir('./dataset/{}'.format(file))):
        os.makedirs(os.path.join('./dataset/{}'.format(file)))

    path = 'dataset/'+file+'/unk_graph.pickle'
    if os.path.exists(path):
        with open(path, 'rb') as handle:
            datasets_unk_graph = pickle.load(handle)
    else:
        path = 'dataset/'+file+'.csv'
        if os.path.exists(path):
            df = pd.read_csv('dataset/'+file+'.csv')
            X = np.array(df[smiles])
            y = np.zeros(len(df[smiles]))
            if 'Id' in df.columns:
                ids = np.array(df['Id'])
            else:
                ids = X # assign ids as original smiles
            datasets = (X, y, ids)
        else:
            print('ERROR: file does not exist.')

        datasets_unk_graph = dict()
        for d in zip(datasets[0], datasets[1], datasets[2]):
            smiles = d[0]
            g = generateGraph(d, with_id=True)
            if g:
                datasets_unk_graph[smiles] = g

        path = 'dataset/'+file+'/unk_graph.pickle'
        # write dataset to pickle 
        with open(path, 'wb') as handle:
            pickle.dump(datasets_unk_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        print('Function: generateGraphDatasetUnknown()')
        print('Datasets Test: ', len(datasets_unk_graph))
    except:
        print('Cannot print dataset statistics')

    return datasets_unk_graph

# generate vocaburary using train data of graph datasets
def generateVocabTrain(file, splitting_seed, splitting_fold=1, vocab_len=100):
    for i in range(splitting_fold):
        print('Generating vocab fold', i)
        path = 'dataset/'+file+'/train_'+str(splitting_seed)+'_'+str(i)+'.csv'
        if os.path.exists(path):
            df = readFromCSV(path)
        smiles_list = list(df['X'])
        vocab_file = file+'_'+str(i)
        vocab_path = 'vocab/'+vocab_file+'.txt'
        if os.path.exists(vocab_path):
            print('Vocab files already exist')
            return
        generate_vocab(smiles_list, vocab_len, vocab_path)

# generate tokenizer using vocab path
def generateTokenizer(vocab_file):
    vocab_path = 'vocab/'+vocab_file+'.txt'
    tokenizer = Tokenizer(vocab_path)
    return tokenizer

# generate reduced graph dict
def generateReducedGraphDict(file, reducedgraph, vocab_file=None):
    if reducedgraph == 'substructure':
        path = 'dataset/'+file+'/'+reducedgraph+'_'+vocab_file+'_final.pickle'
    else:
        path = 'dataset/'+file+'/'+reducedgraph+'_final.pickle'

    if os.path.exists(path):
        with open(path, 'rb') as handle:
            reduced_graph_dict = pickle.load(handle)
    else:
        reduced_graph_dict = {}
        duplicated = []
        tokenizer = generateTokenizer(vocab_file) if reducedgraph == 'substructure' else None

        # read validated if exists
        path_validate = 'dataset/'+file+'/'+file+'_validated.csv'
        if os.path.exists(path_validate):
            datasets = readFromCSV(path_validate)
            datasets = datasets[datasets['v']=='valid']
        X = datasets['X'].values
        y = datasets['y'].values

        for X, y in tqdm(zip(X, y)):
            smiles = X
            d, cliques = constructReducedGraph(reducedgraph, smiles, y, tokenizer)
            if d:
                if smiles in reduced_graph_dict: 
                    duplicated.append(smiles)
                reduced_graph_dict[smiles] = (d, cliques)

        print("number of duplicated smiles:", len(duplicated), duplicated)

        with open(path, 'wb') as handle:
            pickle.dump(reduced_graph_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print('Function: generateReducedGraphDict()')
    # print("number of all reduced graphs:", len(reduced_graph_dict))

    return reduced_graph_dict

# generate data loader 
def generateDataLoader(file, batch_size, seed, fold_number):
    path = 'dataset/'+file+'/graph.pickle'
    if os.path.exists(path):
        with open(path, 'rb') as handle:
            datasets_graph = pickle.load(handle)

    path_train = 'dataset/'+file+'/train_'+str(seed)+'_'+str(fold_number)+'.csv'
    datasets_train_df = readFromCSV(path_train)
    path_val = 'dataset/'+file+'/val_'+str(seed)+'_'+str(fold_number)+'.csv'
    datasets_val_df = readFromCSV(path_val)
    path_test = 'dataset/'+file+'/test.csv'
    datasets_test_df = readFromCSV(path_test)
    
    # datasets_train = [x for x in datasets_graph if x.smiles in datasets_train_df['X'].values] 
    # datasets_val = [x for x in datasets_graph if x.smiles in datasets_val_df['X'].values] 
    # datasets_test = [x for x in datasets_graph if x.smiles in datasets_test_df['X'].values] 
    datasets_train = [datasets_graph[x] for x in datasets_train_df['X'].values if x in datasets_graph] 
    datasets_val = [datasets_graph[x] for x in datasets_val_df['X'].values if x in datasets_graph] 
    datasets_test = [datasets_graph[x] for x in datasets_test_df['X'].values if x in datasets_graph] 
    loader_train = DataLoader(datasets_train, batch_size=batch_size, shuffle=True, follow_batch=['x_g'])
    loader_val = DataLoader(datasets_val, batch_size=batch_size, shuffle=False, follow_batch=['x_g'])
    loader_test = DataLoader(datasets_test, batch_size=batch_size, shuffle=False, follow_batch=['x_g'])
    return loader_train, loader_val, loader_test, datasets_train, datasets_val, datasets_test

# generate data loader for Testing
def generateDataLoaderTesting(file, batch_size):
    path = 'dataset/'+file+'/graph.pickle'
    if os.path.exists(path):
        with open(path, 'rb') as handle:
            datasets_graph = pickle.load(handle)

    path_test = 'dataset/'+file+'/test.csv'
    datasets_test_df = readFromCSV(path_test)

    # datasets_test = [x for x in datasets_graph if x.smiles in datasets_test_df['X'].values] 
    datasets_test = [datasets_graph[x] for x in datasets_test_df['X'].values if x in datasets_graph] 
    loader_test = DataLoader(datasets_test, batch_size=batch_size, shuffle=False, follow_batch=['x_g'])
    return loader_test, datasets_test

# generate data loader for training+val
def generateDataLoaderTraining(file, batch_size):
    path = 'dataset/'+file+'/graph.pickle'
    if os.path.exists(path):
        with open(path, 'rb') as handle:
            datasets_graph = pickle.load(handle)

    path_trainval = 'dataset/'+file+'/'+file+'_validated.csv'
    datasets_trainval_df = readFromCSV(path_trainval)

    # datasets_trainval = [x for x in datasets_graph if x.smiles in datasets_trainval_df['X'].values] 
    datasets_trainval = [datasets_graph[x] for x in datasets_trainval_df['X'].values if x in datasets_graph] 
    loader_trainval = DataLoader(datasets_trainval, batch_size=batch_size, shuffle=False, follow_batch=['x_g'])
    return loader_trainval, datasets_trainval

# generate data loader for list of graph data
def generateDataLoaderListing(datasets_list, batch_size):
    loader_test = DataLoader(datasets_list, batch_size=batch_size, shuffle=False, follow_batch=['x_g'])
    return loader_test, datasets_list

# get number of smiles in item
def getNumberofSmiles(item):
    return len(item.smiles)

