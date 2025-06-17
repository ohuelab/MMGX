######################
### Import Library ###
######################

# my library
from locale import normalize
from molgraph.utilsonehot import *
from molgraph.utilsmol import *
from molgraph.utilsgraph import *
# standard
import os as os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
# rdkit
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import FragmentCatalog
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
# deepchem
import deepchem.feat as dcfeat

#########################
### Global Definition ###
#########################

# for junction tree and cluster
MST_MAX_WEIGHT = 100 

# for every graph
definedAtom = [
    'C','N','O','S','F','Si','P','Cl','Br','Mg',
    'Na','Ca','Fe','As','Al','I','B','V','K','Tl',
    'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', # H?
    'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr',
    'Pt','Hg','Pb',
    'Unknown'
]
NUMBER_OF_ATOM = len(definedAtom)

definedBond = [
    Chem.rdchem.BondType.SINGLE, 
    Chem.rdchem.BondType.DOUBLE, 
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]
NUMBER_OF_BOND = len(definedBond)

# for functional
fName = os.path.join(RDConfig.RDDataDir,'FunctionalGroups.txt')
fparams = FragmentCatalog.FragCatParams(1, 6, fName)
NUMBER_OF_FUNCTIONAL = len(range(fparams.GetNumFuncGroups()))

definedFuncBond = [
    ('C', Chem.rdchem.BondType.SINGLE, 'C'),
    ('C', Chem.rdchem.BondType.DOUBLE, 'C'),
    ('C', Chem.rdchem.BondType.TRIPLE, 'C'),
    ('C', Chem.rdchem.BondType.SINGLE, 'O'),
    ('C', Chem.rdchem.BondType.DOUBLE, 'O'),
    ('C', Chem.rdchem.BondType.SINGLE, 'N'),
    ('C', Chem.rdchem.BondType.DOUBLE, 'N'),
    ('C', Chem.rdchem.BondType.TRIPLE, 'N'),
    ('C', Chem.rdchem.BondType.SINGLE, 'S'),
    ('C', Chem.rdchem.BondType.DOUBLE, 'S'),
    ('C', Chem.rdchem.BondType.TRIPLE, 'S'),
    ('O', Chem.rdchem.BondType.SINGLE, 'O'),
    ('O', Chem.rdchem.BondType.SINGLE, 'N'),
    ('O', Chem.rdchem.BondType.DOUBLE, 'N'),
    ('O', Chem.rdchem.BondType.SINGLE, 'S'),
    ('N', Chem.rdchem.BondType.SINGLE, 'N'),
    ('N', Chem.rdchem.BondType.DOUBLE, 'N'),
    ('N', Chem.rdchem.BondType.SINGLE, 'S'),
    ('N', Chem.rdchem.BondType.TRIPLE, 'S'),
    ('N', Chem.rdchem.BondType.DOUBLE, 'S'),
    ('S', Chem.rdchem.BondType.SINGLE, 'S'),
    ('S', Chem.rdchem.BondType.DOUBLE, 'S'),
    ('S', Chem.rdchem.BondType.TRIPLE, 'S'),
    'Unknown'
]
NUMBER_OF_FUNCBOND = len(definedFuncBond)

fring = pd.read_csv('util/func_ring.txt', sep='\t')
definedRing = list(fring['smarts'])
NUMBER_OF_FUNCRING = len(definedRing)

##########################
### Utilities Function ###
##########################

# print global definition
def printGlobalDefinition():
    print('ATOM', NUMBER_OF_ATOM)
    for (i, item) in enumerate(definedAtom):
        print(i, item)
    print('BOND', NUMBER_OF_BOND)
    for (i, item) in enumerate(definedBond):
        print(i, item)
    print('FUNCTIONAL', NUMBER_OF_FUNCTIONAL)
    for i in range(fparams.GetNumFuncGroups()):
        if i == 27:
            print('SWAP WITH 29')
        if i == 29:
            print('SWAP WITH 27')
        print(i, fparams.GetFuncGroup(i).GetProp('_Name'), Chem.MolToSmarts(fparams.GetFuncGroup(i)))
    print('FUNC_BOND', NUMBER_OF_FUNCBOND)
    for (i, item) in enumerate(definedFuncBond):
        print(i, item)
    print('FUNC_RING', NUMBER_OF_FUNCRING)
    print(fring)

############################
### Class Initialization ###
############################

# MoleculeGraph
class MoleculeGraph:
    def __init__(self, smiles):
        self.smiles = mol_to_smiles(smiles_to_mol(smiles))
        self.mol = smiles_to_mol(self.smiles)
        self.cliques = None
        self.edges = None
        self.graph_size = 0
        self.node_index = []
        self.node_features = []
        self.edge_index = []
        self.edge_features = []

    def getMoleculeGraph(self):
        return self.mol, self.smiles, self.cliques, self.edges, self.graph_size, \
            self.node_index, self.node_features, self.edge_index, self.edge_features,

    def __str__(self):
        if not isinstance(self, AtomGraph):
            return "Molecule Graph: "+self.smiles+"\n"+ \
                "Node Index:"+str(self.node_index)+"\n"+ \
                "Edge Index:"+str(self.edge_index)+"\n"+ \
                "Cliques:"+str(self.cliques)
        else:
            return "Molecule Graph: "+self.smiles+"\n"+ \
                "Node Index:"+str(self.node_index)+"\n"+ \
                "Edge Index:"+str(self.edge_index)

# ATOM-based
class AtomGraph(MoleculeGraph):
    def __init__(self, smiles, normalize=True):
        self.smiles = mol_to_smiles(smiles_to_mol(smiles))
        self.mol = smiles_to_mol(self.smiles)
        self.normalize =normalize
        graph_size, node_index, node_attr, edge_index, edge_attr = self.mol_to_graph(self.mol, normalize)
        self.graph_size = graph_size
        self.node_index = node_index
        self.node_features = node_attr
        self.edge_index = edge_index
        self.edge_features = edge_attr
        # for only atom graph
        self.cliques = [[i] for i in node_index]
        self.edges = edge_index

    def getAtomFeature(self, atom):
        # number of deepchem features (78)
        atom_features = list(np.multiply(dcfeat.graph_features.atom_features(atom, use_chirality=True), 1))
        # additional features (79)
        atom_features += [np.multiply(atom.IsInRing(), 1)]
        return atom_features

    def getBondFeature(self, bond):
        # number of deepchem features (10)
        bond_features = list(np.multiply(dcfeat.graph_features.bond_features(bond, use_chirality=True), 1))
        return bond_features

    def mol_to_graph(self, mol, normalize=True):
        graph_size = mol.GetNumAtoms()
        
        node_attr = [] 
        node_index = []
        for atom in mol.GetAtoms():
            node_feature = self.getAtomFeature(atom)
            if normalize:
                node_attr.append(list(node_feature)/sum(node_feature))
            else:
                node_attr.append(list(node_feature))
            node_index.append(atom.GetIdx())

        edge_attr = []
        edge_index = []
        for bond in mol.GetBonds():
            edge_feature = self.getBondFeature(bond)
            # from 1 -> 2
            if normalize and sum(edge_feature) != 0:
                edge_attr.append(list(edge_feature)/sum(edge_feature))
            else:
                edge_attr.append(list(edge_feature))
            edge_index.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            # from 2 -> 1
            if normalize and sum(edge_feature) != 0:
                edge_attr.append(list(edge_feature)/sum(edge_feature))
            else:
                edge_attr.append(list(edge_feature))
            edge_index.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
            
        return graph_size, node_index, node_attr, edge_index, edge_attr

# JUNCTIONTREE-based
# ref: https://arxiv.org/pdf/1802.04364.pdf
class JunctionTreeGraph(MoleculeGraph):
    def __init__(self, smiles, normalize=True):
        self.smiles = mol_to_smiles(smiles_to_mol(smiles))
        self.mol = smiles_to_mol(self.smiles)
        self.normalize =normalize
        cliques, edges  = self.getReducedGraph(self.mol)
        self.cliques = cliques
        self.edges = edges
        clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_graph(self.mol, cliques, edges, normalize)
        self.graph_size = clique_size
        self.node_index = clique_index
        self.node_features = clique_attr
        self.edge_index = cliqueedge_index
        self.edge_features = cliqueedge_attr

    # form a junction tree graph with intersection as a node
    #      A                AC
    #      |      >>>       |     
    #  B - C - D       BC - C - DC
    def getReducedGraph(self, mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1,a2])

        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)

        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)
        
        # Merge rings that share more than 2 atoms as they form bridged compounds.
        for i in range(len(cliques)):
            if len(cliques[i]) <= 2: continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2: continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []
        
        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)
        
        # Build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1: 
                continue
            cnei = nei_list[atom]
            # Number of bond clusters that the atom lies in.
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            # Number of ring clusters that the atom lies in.
            rings = [c for c in cnei if len(cliques[c]) > 4]
            # In general, if len(cnei) >= 3, a singleton should be added, 
            # but 1 bond + 2 ring is currently not dealt with.
            if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): 
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1,c2)] = 1
            # at least 1 bond connect to at least 2 rings
            elif len(rings) >= 2 and len(bonds) >= 1:
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1,c2)] = MST_MAX_WEIGHT - 1
            # Multiple (n>2) complex rings
            elif len(rings) > 2: 
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1,c2)] = MST_MAX_WEIGHT - 1 
            else:
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1,c2 = cnei[i],cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1,c2)] < len(inter):
                            # cnei[i] < cnei[j] by construction ?
                            edges[(c1,c2)] = len(inter) 
                            edges[(c2,c1)] = len(inter) 

        # check isolated single atom 
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in nei_list[atom.GetIdx()] and len(atom.GetBonds()) == 0:
                cliques.append([atom.GetIdx()])
                nei_list[atom.GetIdx()].append(len(cliques)-1)

        edges = [u + (MST_MAX_WEIGHT-v,) for u,v in edges.items()]
        if len(edges) == 0:
            return cliques, edges

        # Compute Maximum Spanning Tree
        row,col,data = zip(*edges)
        n_clique = len(cliques)
        clique_graph = csr_matrix((data,(row,col)), shape=(n_clique,n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row,col = junc_tree.nonzero()
        # edges = [(row[i],col[i]) for i in range(len(row))]
        edges = []
        for i in range(len(row)):
            edges.append((row[i],col[i]))
            edges.append((col[i],row[i]))
        return cliques, edges  

    def getCliqueFeatures(self, clique, edges, clique_idx, mol):
        # number of node features (83)
        NumEachAtomDict = {a: 0 for a in definedAtom}
        NumEachBondDict = {b: 0 for b in definedBond}
        
        # number of atoms
        NumAtoms = len(clique)
        # number of edges 
        NumEdges = 0  
        for edge in edges:
            if clique_idx == edge[0] or clique_idx == edge[1]:
                NumEdges += 1
        # number of Hs
        NumHs = 0 
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in clique:
                NumHs += atom.GetTotalNumHs()
                # number of each atom
                sb = atom.GetSymbol()
                if sb in NumEachAtomDict:
                    NumEachAtomDict[sb] += 1
                else:
                    NumEachAtomDict['Unknown'] += 1
        # is ring
        IsRing = 0
        if len(clique) > 2:
            IsRing = 1
        # is bond
        IsBond = 0
        if len(clique) == 2:
            IsBond = 1
        # number of each bond
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in clique and bond.GetEndAtomIdx() in clique:
                bt = bond.GetBondType()
                NumEachBondDict[bt] += 1 
        
        # convert number of each atom
        if sum(list(NumEachAtomDict.values())) != 0 and self.normalize:
            NumEachAtom = [float(i)/sum(list(NumEachAtomDict.values())) for i in list(NumEachAtomDict.values())] 
        else:
            NumEachAtom = [int(i) for i in list(NumEachAtomDict.values())] 
        # convert number of each bond
        if sum(list(NumEachBondDict.values())) != 0 and self.normalize:
            NumEachBond = [float(i)/sum(list(NumEachBondDict.values())) for i in list(NumEachBondDict.values())] 
        else:
            NumEachBond = [int(i) for i in list(NumEachBondDict.values())] 

        return np.array(
            one_of_k_encoding_unk(NumAtoms,[0,1,2,3,4,5,6,7,8,9,10]) + 
            one_of_k_encoding_unk(NumEdges,[0,1,2,3,4,5,6,7,8,9,10]) + 
            one_of_k_encoding_unk(NumHs,[0,1,2,3,4,5,6,7,8,9,10]) + 
            [IsRing] + 
            [IsBond] +
            NumEachAtom +
            NumEachBond
            )

    def getCliqueEdgeFeatures(self, edge, clique, idx, mol):
        # number of edge features (6)
        len_begin = len(clique[edge[0]])
        len_end = len(clique[edge[1]])
        
        EdgeType = 5
        if len_begin == 1:
            begin_type = 'atom'
        elif len_begin > 2:
            begin_type = 'ring'
        else: 
            begin_type = 'bond'
        if len_end == 1:
            end_type = 'atom'
        elif len_end > 2:
            end_type = 'ring'
        else: 
            end_type = 'bond'
            
        definedEdgeType = {('atom', 'atom'): 0,
                        ('atom', 'bond'): 1,
                        ('atom', 'ring'): 2,
                        ('bond', 'atom'): 1,
                        ('bond', 'bond'): 3,
                        ('bond', 'ring'): 4,
                        ('ring', 'atom'): 2,
                        ('ring', 'bond'): 4,
                        ('ring', 'ring'): 5}
        EdgeType = definedEdgeType[(begin_type, end_type)]
        
        return np.array(one_of_k_encoding(EdgeType,list(set(definedEdgeType.values()))))

    def mol_to_graph(self, mol, cliques, edges, normalize=True):
        clique_size = len(cliques)
        
        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            clique_features = self.getCliqueFeatures(cliques[idx], edges, idx, mol)
            if normalize and sum(clique_features) != 0:
                clique_attr.append(list(clique_features)/sum(clique_features))
            else:
                clique_attr.append(clique_features)
        
        cliqueedge_index = edges
        cliqueedge_attr = []
        for idx in range(len(edges)):
            cliqueedge_features = self.getCliqueEdgeFeatures(edges[idx], cliques, idx, mol)
            if normalize and sum(cliqueedge_features) != 0:
                cliqueedge_attr.append(list(cliqueedge_features)/sum(cliqueedge_features))
            else:
                cliqueedge_attr.append(cliqueedge_features)

        return clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr

# CLUSTER-based
class ClusterGraph(MoleculeGraph):
    def __init__(self, smiles, normalize=True):
        self.smiles = mol_to_smiles(smiles_to_mol(smiles))
        self.mol = smiles_to_mol(self.smiles)
        self.normalize = normalize
        cliques, edges  = self.getReducedGraph(self.mol)
        self.cliques = cliques
        self.edges = edges
        clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_graph(self.mol, cliques, edges, normalize)
        self.graph_size = clique_size
        self.node_index = clique_index
        self.node_features = clique_attr
        self.edge_index = cliqueedge_index
        self.edge_features = cliqueedge_attr

    # form a cluster graph with intersection as a cycle
    #      A               AC
    #      |      >>>    /    \   
    #  B - C - D       BC ---- DC
    def getReducedGraph(self, mol):
        n_atoms = mol.GetNumAtoms()

        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1,a2])

        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)

        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # # Merge rings that share more than 2 atoms as they form bridged compounds.
        # for i in range(len(cliques)):
        #     if len(cliques[i]) <= 2: continue
        #     for atom in cliques[i]:
        #         for j in nei_list[atom]:
        #             if i >= j or len(cliques[j]) <= 2: 
        #                 continue
        #             inter = set(cliques[i]) & set(cliques[j])
        #             if len(inter) > 2:
        #                 cliques[i].extend(cliques[j])
        #                 cliques[i] = list(set(cliques[i]))
        #                 cliques[j] = []
        
        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1: 
                continue
            cnei = nei_list[atom]
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            rings = [c for c in cnei if len(cliques[c]) > 4]
            
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1,c2 = cnei[i],cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):
                        # cnei[i] < cnei[j] by construction ?
                        edges[(c1,c2)] = len(inter) 
                        edges[(c2,c1)] = len(inter)

        # check isolated single atom 
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in nei_list[atom.GetIdx()] and len(atom.GetBonds()) == 0:
                cliques.append([atom.GetIdx()])
                nei_list[atom.GetIdx()].append(len(cliques)-1)

        edges = [e for e in edges]
        if len(edges) == 0:
            return cliques, edges

        # row,col,data = zip(*edges)
        # data = list(data)
        # for i in range(len(data)):
        #     data[i] = 1
        # data = tuple(data)
        # n_clique = len(cliques)
        # clique_graph = csr_matrix((data,(row,col)), shape=(n_clique,n_clique))
        
        return cliques, edges

    def getCliqueFeatures(self, clique, edges, clique_idx, mol):
        # number of node features (83)
        NumEachAtomDict = {a: 0 for a in definedAtom}
        NumEachBondDict = {b: 0 for b in definedBond}
        
        # number of atoms
        NumAtoms = len(clique)
        # number of edges 
        NumEdges = 0  
        for edge in edges:
            if clique_idx == edge[0] or clique_idx == edge[1]:
                NumEdges += 1
        # number of Hs
        NumHs = 0 
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in clique:
                NumHs += atom.GetTotalNumHs()
                # number of each atom
                sb = atom.GetSymbol()
                if sb in NumEachAtomDict:
                    NumEachAtomDict[sb] += 1
                else:
                    NumEachAtomDict['Unknown'] += 1
        # is ring
        IsRing = 0
        if len(clique) > 2:
            IsRing = 1
        # is bond
        IsBond = 0
        if len(clique) == 2:
            IsBond = 1
        # number of each bond
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in clique and bond.GetEndAtomIdx() in clique:
                bt = bond.GetBondType()
                NumEachBondDict[bt] += 1 
        
        # convert number of each atom
        if sum(list(NumEachAtomDict.values())) != 0 and self.normalize:
            NumEachAtom = [float(i)/sum(list(NumEachAtomDict.values())) for i in list(NumEachAtomDict.values())] 
        else:
            NumEachAtom = [int(i) for i in list(NumEachAtomDict.values())] 
        # convert number of each bond
        if sum(list(NumEachBondDict.values())) != 0 and self.normalize:
            NumEachBond = [float(i)/sum(list(NumEachBondDict.values())) for i in list(NumEachBondDict.values())] 
        else:
            NumEachBond = [int(i) for i in list(NumEachBondDict.values())] 

        return np.array(
            one_of_k_encoding_unk(NumAtoms,[0,1,2,3,4,5,6,7,8,9,10]) + 
            one_of_k_encoding_unk(NumEdges,[0,1,2,3,4,5,6,7,8,9,10]) + 
            one_of_k_encoding_unk(NumHs,[0,1,2,3,4,5,6,7,8,9,10]) + 
            [IsRing] + 
            [IsBond] +
            NumEachAtom +
            NumEachBond
            )

    def getCliqueEdgeFeatures(self, edge, clique, idx, mol):
        # number of edge features (6)
        len_begin = len(clique[edge[0]])
        len_end = len(clique[edge[1]])
        
        EdgeType = 5
        if len_begin == 1:
            begin_type = 'atom'
        elif len_begin > 2:
            begin_type = 'ring'
        else: 
            begin_type = 'bond'
        if len_end == 1:
            end_type = 'atom'
        elif len_end > 2:
            end_type = 'ring'
        else: 
            end_type = 'bond'
            
        definedEdgeType = {('atom', 'atom'): 0,
                        ('atom', 'bond'): 1,
                        ('atom', 'ring'): 2,
                        ('bond', 'atom'): 1,
                        ('bond', 'bond'): 3,
                        ('bond', 'ring'): 4,
                        ('ring', 'atom'): 2,
                        ('ring', 'bond'): 4,
                        ('ring', 'ring'): 5}
        EdgeType = definedEdgeType[(begin_type, end_type)]
        
        return np.array(one_of_k_encoding(EdgeType,list(set(definedEdgeType.values()))))

    def mol_to_graph(self, mol, cliques, edges, normalize=True):
        clique_size = len(cliques)
        
        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            clique_features = self.getCliqueFeatures(cliques[idx], edges, idx, mol)
            if normalize and sum(clique_features) != 0:
                clique_attr.append(list(clique_features)/sum(clique_features))
            else:
                clique_attr.append(clique_features)
        
        cliqueedge_index = edges
        cliqueedge_attr = []
        for idx in range(len(edges)):
            cliqueedge_features = self.getCliqueEdgeFeatures(edges[idx], cliques, idx, mol)
            if normalize and sum(cliqueedge_features) != 0:
                cliqueedge_attr.append(list(cliqueedge_features)/sum(cliqueedge_features))
            else:
                cliqueedge_attr.append(cliqueedge_features)

        return clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr

# FUNCTIONAL-based
# ref: https://github.com/rdkit/rdkit/blob/master/Data/FunctionalGroups.txt
class FunctionalGraph(MoleculeGraph):
    def __init__(self, smiles, normalize=True):
        self.smiles = mol_to_smiles(smiles_to_mol(smiles))
        self.mol = smiles_to_mol(self.smiles)
        self.normalize =normalize
        cliques, edges, cliques_func, cliques_ring = self.getReducedGraph(self.mol)
        self.cliques = cliques
        self.edges = edges
        self.cliques_func = cliques_func
        self.cliques_ring = cliques_ring
        clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_graph(self.mol, cliques, edges, cliques_func, cliques_ring, normalize)
        self.graph_size = clique_size
        self.node_index = clique_index
        self.node_features = clique_attr
        self.edge_index = cliqueedge_index
        self.edge_features = cliqueedge_attr

    def getReducedGraph(self, mol):
        n_atoms = mol.GetNumAtoms()

        # functional group
        funcGroupDict = dict()
        for i in range(fparams.GetNumFuncGroups()):
            funcGroupDict[i] = list(mol.GetSubstructMatches(fparams.GetFuncGroup(i)))
            
        # edit #27 <-> #29
        temp = funcGroupDict[27]
        funcGroupDict[27] = funcGroupDict[29]
        funcGroupDict[29] = temp

        cliques = []
        cliques_ring = {} # node group in ring
        cliques_func = {} # node group in func
        seen_func = {} # node seen in func
        group_num = 0
        
        # extract functional from substructure match
        for f in funcGroupDict:
            for l in funcGroupDict[f]:
                if not(all(ll in seen_func for ll in l)):
                    cliques.append(list(l))
                    for ll in l:
                        if ll in seen_func or ll in cliques_func:
                            cliques_func[ll].append(f)
                            seen_func[ll].append(group_num)
                        else:
                            cliques_func[ll] = [f]
                            seen_func[ll] = [group_num]
                group_num += 1

        # extract bond which not in functional
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            if not bond.IsInRing():
                if (a1 not in seen_func) or (a2 not in seen_func):
                    cliques.append([a1,a2])
                elif a1 in seen_func and a2 in seen_func and len(set(seen_func[a1]) & set(seen_func[a2]))==0:
                    cliques.append([a1,a2])

        # extract ring        
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)
        cliques_ring = ssr
        
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # # Merge rings that share more than 2 atoms as they form bridged compounds.
        # for i in range(len(cliques)):
        #     if len(cliques[i]) <= 2: continue
        #     for atom in cliques[i]:
        #         for j in nei_list[atom]:
        #             if i >= j or len(cliques[j]) <= 2: 
        #                 continue
        #             inter = set(cliques[i]) & set(cliques[j])
        #             if len(inter) > 2:
        #                 cliques[i].extend(cliques[j])
        #                 cliques[i] = list(set(cliques[i]))
        #                 cliques[j] = []

        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1: 
                continue
            cnei = nei_list[atom]
            # Number of bond clusters that the atom lies in.
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            # Number of ring clusters that the atom lies in.
            funcring = [c for c in cnei if len(cliques[c]) > 2]

            # # In general, if len(cnei) >= 3, a singleton should be added, 
            # if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): 
            #     cliques.append([atom])
            #     c2 = len(cliques) - 1
            #     for c1 in cnei:
            #         edges[(c1,c2)] = 1
            # # at least 1 bond connect to at least 2 funcring
            # elif len(funcring) >= 2 and len(bonds) >= 1:
            #     cliques.append([atom])
            #     c2 = len(cliques) - 1
            #     for c1 in cnei:
            #         edges[(c1,c2)] = MST_MAX_WEIGHT - 1
            # # Multiple (n>2) complex funcring
            # elif len(funcring) > 2: 
            #     cliques.append([atom])
            #     c2 = len(cliques) - 1
            #     for c1 in cnei:
            #         edges[(c1,c2)] = MST_MAX_WEIGHT - 1 
            # else:
            #     for i in range(len(cnei)):
            #         for j in range(i + 1, len(cnei)):
            #             c1,c2 = cnei[i],cnei[j]
            #             inter = set(cliques[c1]) & set(cliques[c2])
            #             if edges[(c1,c2)] < len(inter):
            #                 # cnei[i] < cnei[j] by construction ?
            #                 edges[(c1,c2)] = len(inter) 
            #                 edges[(c2,c1)] = len(inter)

            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1,c2 = cnei[i],cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):
                        # cnei[i] < cnei[j] by construction ?
                        edges[(c1,c2)] = len(inter) 
                        edges[(c2,c1)] = len(inter)

        # check isolated single atom 
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in nei_list[atom.GetIdx()] and len(atom.GetBonds()) == 0:
                cliques.append([atom.GetIdx()])
                nei_list[atom.GetIdx()].append(len(cliques)-1)
        
        edges = [i for i in edges]
        # print('1', edges)
        # edges = [u + (MST_MAX_WEIGHT-v,) for u,v in edges.items()]
        # print('2', edges)
        # if len(edges) == 0:
        #     return cliques, edges, cliques_func, cliques_ring
        
        # row, col, data = zip(*edges)
        # # data = list(data)
        # # for i in range(len(data)):
        # #     data[i] = 1
        # # data = tuple(data)
        # # n_clique = len(cliques)
        # # clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
        # edges = [(row[i],col[i]) for i in range(len(row))]
        # print('3', edges)

        return cliques, edges, cliques_func, cliques_ring


    def getCliqueFeatures_funcGroup(self, clique, edges, clique_idx, cliques_func, cliques_ring, mol):
        # number of node features (115)
        funcType = [0 for f in range(len(range(fparams.GetNumFuncGroups())))]  # no unknown
        funcRingTypeList = range(len(definedRing)) # (only aromatic)
        funcRingTypeOtherList = range(len(definedRing)) # (other bonds)
        funcRingTypeSizeList = [3,4,5,6,7,8,9,10] # unknown ring size 3-9 and >9
        funcBondTypeList = range(len(definedFuncBond)) # included unknown
        # atomTypeList = range(len(definedAtom)) # included unknown
        
        ringType = one_of_k_encoding_none(None, funcRingTypeList)
        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
        funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
        # atomType = atomType = one_of_k_encoding_none(None, atomTypeList)

        # functional group
        func_found = False
        if all(c in cliques_func for c in clique):
            funcGroup = [cliques_func[c] for c in clique]
            intersect = funcGroup[0]
            for f in funcGroup:
                intersect = set(set(intersect) & set(f))
            for i in list(intersect):
                funcType[i] = 1
                func_found = True
            if func_found: # func, not ring not bond
                ringType = one_of_k_encoding_none(None, funcRingTypeList)
                ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
        # ring type
        if len(clique) > 2 and not func_found:
            if clique in cliques_ring:
                new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol)) # not kekulize
                smarts = Chem.MolFragmentToSmarts(new_mol, clique)
                ring_found = False
                for ring in definedRing:
                    mol_ring = Chem.MolFromSmarts(ring)
                    mol_smart = Chem.MolFromSmarts(smarts)
                    t1 = topology_checker(mol_ring)
                    t2 = topology_checker(mol_smart)
                    if len(mol_smart.GetSubstructMatches(mol_ring))!=0:
                        ringType = one_of_k_encoding_unk(definedRing.index(ring), funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
                        ring_found = True
                        break
                    elif is_isomorphic(t1, t2):
                        ringType = one_of_k_encoding_unk(definedRing.index(ring), funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
                        ring_found = True
                        break
                    elif is_isomorphic_atom(t1, t2):
                        ringType = one_of_k_encoding_none(None, funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_unk(definedRing.index(ring), funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
                        ring_found = True
                        break
                if not ring_found: # unknown ring
                    mol_smart = Chem.MolFromSmarts(smarts)
                    ringType = one_of_k_encoding_none(None, funcRingTypeList)
                    ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                    ringTypeSize = one_of_k_encoding_unk(mol_smart.GetNumAtoms(), funcRingTypeSizeList)
                    funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
            else: # not ring not bond
                ringType = one_of_k_encoding_none(None, funcRingTypeList)
                ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
        # bond type
        if len(clique) == 2 and not func_found:
            bond_found = False
            for bond in mol.GetBonds():
                b = bond.GetBondType()
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                a1 = atom1.GetIdx()
                a2 = atom2.GetIdx()
                a1_s = atom1.GetSymbol()
                a2_s = atom2.GetSymbol()
                if [a1, a2] == clique or [a2, a1] == clique:
                    if (a1_s, b, a2_s) in definedFuncBond:
                        funcBondType = one_of_k_encoding_unk(definedFuncBond.index((a1_s, b, a2_s)), funcBondTypeList)
                        ringType = one_of_k_encoding_none(None, funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        bond_found = True
                        break
                    elif (a2_s, b, a1_s) in definedFuncBond:
                        funcBondType = one_of_k_encoding_unk(definedFuncBond.index((a2_s, b, a1_s)), funcBondTypeList)
                        ringType = one_of_k_encoding_none(None, funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        bond_found = True
                        break
            if not bond_found: # unknown bond
                ringType = one_of_k_encoding_none(None, funcRingTypeList)
                ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                funcBondType = one_of_k_encoding_unk(None, funcBondTypeList)
                    
        # # some cliques are one atom and be a part of func
        # if len(clique) == 1:
        #     # atom type
        #     for atom in mol.GetAtoms():
        #         a = atom.GetIdx()
        #         a_s = atom.GetSymbol()
        #         if [a] == clique:
        #             if a_s in definedAtom:
        #                 atomType = one_of_k_encoding_unk(definedAtom.index(a_s), atomTypeList)
        # else:
        #     atomType = one_of_k_encoding_none(None, atomTypeList)

        # return np.array(funcType+ringType+funcBondType+atomType)
        return np.array(funcType+ringType+ringTypeOther+ringTypeSize+funcBondType)


    def getCliqueEdgeFeatures_funcGroup(self, edge, clique, edge_idx, cliques_func, cliques_ring, mol):
        # number of edge features (10)+10 = (20) # (10)+12 = (22)
        begin = clique[edge[0]]
        end = clique[edge[1]]
        
        if all(c in cliques_func for c in begin):
            begin_type = 'func'
        elif len(begin) > 2:
            begin_type = 'ring'
        elif len(begin) == 1:
            begin_type = 'atom'
        else:
            begin_type = 'bond'
            
        if all(c in cliques_func for c in end):
            end_type = 'func'
        elif len(end) > 2:
            end_type = 'ring'
        elif len(end) == 1:
            end_type = 'atom'
        else:
            end_type = 'bond'

        intersect = len(set(begin) & set(end))

        begin_atom = 0
        end_atom = 0
        if intersect == 1:
            a1 = list(set(begin) & set(end))[0]
            a2 = list(set(begin) & set(end))[0]
            begin_atom = sorted(begin).index(a1)+1
            end_atom = sorted(end).index(a2)+1
        # in case, more than 2 common atoms
        else:
            begin_atom = 0
            end_atom = 0 
        
        definedEdgeType = {('atom', 'atom'): 0,
                        ('atom', 'bond'): 1,
                        ('atom', 'ring'): 2,
                        ('atom', 'func'): 3,
                        ('bond', 'atom'): 1,
                        ('bond', 'bond'): 4,
                        ('bond', 'ring'): 5,
                        ('bond', 'func'): 6,
                        ('ring', 'atom'): 2,
                        ('ring', 'bond'): 5,
                        ('ring', 'ring'): 7,
                        ('ring', 'func'): 8,
                        ('func', 'atom'): 3,
                        ('func', 'bond'): 6,
                        ('func', 'ring'): 8,
                        ('func', 'func'): 9}
        edgeType = definedEdgeType[(begin_type, end_type)]
        
        # return np.array(one_of_k_encoding(edgeType,list(set(definedEdgeType.values())))+
        #                 one_of_k_encoding_unk(intersect,list(range(10)))+[begin_atom]+[end_atom])
        return np.array(one_of_k_encoding(edgeType,list(set(definedEdgeType.values())))+
                        one_of_k_encoding_unk(intersect,list(range(10))))

    def mol_to_graph(self, mol, cliques, edges, cliques_func, cliques_ring, normalize=True):
        clique_size = len(cliques)

        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            clique_features = self.getCliqueFeatures_funcGroup(cliques[idx], edges, idx, cliques_func, cliques_ring, mol)
            if normalize and sum(clique_features) != 0:
                clique_attr.append(list(clique_features)/sum(clique_features))
            else:
                clique_attr.append(clique_features)
        
        cliqueedge_index = edges
        cliqueedge_attr = []
        for idx in range(len(edges)):
            cliqueedge_features = self.getCliqueEdgeFeatures_funcGroup(edges[idx], cliques, idx, cliques_func, cliques_ring, mol)
            if normalize and sum(cliqueedge_features) != 0:
                cliqueedge_attr.append(list(cliqueedge_features)/sum(cliqueedge_features))
            else:
                cliqueedge_attr.append(cliqueedge_features)
        
        return clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr

# PHARMACOPHORE-based
# ref: https://github.com/rdkit/rdkit/blob/b208da471f8edc88e07c77ed7d7868649ac75100/Code/GraphMol/ReducedGraphs/ReducedGraphs.cpp
class PharmacophoreGraph(MoleculeGraph):
    def __init__(self, smiles, normalize=True):
        self.smiles = mol_to_smiles(smiles_to_mol(smiles))
        self.mol = smiles_to_mol(self.smiles)
        self.normalize =normalize
        cliques, edges, cliques_prop = self.getReducedGraph(self.mol)
        self.cliques = cliques
        self.edges = edges
        self.cliques_prop = cliques_prop
        clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_graph(self.mol, cliques, edges, cliques_prop, normalize)
        self.graph_size = clique_size
        self.node_index = clique_index
        self.node_features = clique_attr
        self.edge_index = cliqueedge_index
        self.edge_features = cliqueedge_attr

    def getReducedGraph(self, mol):
        mol = mol_with_atom_index(mol)
        mol_g = Chem.rdReducedGraphs.GenerateMolExtendedReducedGraph(mol)
        mol_g.UpdatePropertyCache(False)
        mapping_atom = {a.GetAtomMapNum(): i for i, a in enumerate(mol_g.GetAtoms())}
        
        cliques = []
        cliques_prop = []

        ring_8 = [list(x) for x in Chem.GetSymmSSSR(mol) if len(list(x)) < 8]
        ring_B8 = [list(x) for x in Chem.GetSymmSSSR(mol) if len(list(x)) >= 8]

        # add more 8-atom ring
        if len(ring_B8) > 0:
            rwmol_g = Chem.RWMol(mol_g)
            for rb8 in ring_B8:
                new_a = rwmol_g.AddAtom(Chem.Atom(0))
                rwmol_g.GetAtomWithIdx(new_a).SetProp('_ErGAtomTypes', '')
                for rb8_a in rb8:
                    if rb8_a in mapping_atom:
                        rwmol_g.AddBond(new_a, mapping_atom[rb8_a], Chem.BondType.SINGLE)
            mol_g = rwmol_g

        # display(mol_g)
        num_ring_8 = 0   
        num_ring_B8 = 0    

        for atom in mol_g.GetAtoms():
            if atom.GetAtomMapNum() == 0 and atom.GetSymbol() == '*' and 5 in list(atom.GetPropsAsDict()['_ErGAtomTypes']):
                cliques.append(ring_8[num_ring_8])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))
                num_ring_8 += 1
            elif atom.GetAtomMapNum() == 0 and atom.GetSymbol() == '*' and 4 in list(atom.GetPropsAsDict()['_ErGAtomTypes']):
                cliques.append(ring_8[num_ring_8])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))
                num_ring_8 += 1 
            elif atom.GetAtomMapNum() == 0 and atom.GetSymbol() == '*' and atom.IsInRing():
                cliques.append(ring_B8[num_ring_B8])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))
                num_ring_B8 += 1
            else:
                cliques.append([atom.GetAtomMapNum()])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))

        edges = []
        for bond in mol_g.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            edges.append((a1, a2))
            edges.append((a2, a1))
        
        return cliques, edges, cliques_prop

    def getCliqueFeatures_pharmacophore(self, clique, edges, clique_idx, cliques_prop, mol):
        # number of node features (6) 
        pharmacophore = np.zeros(6)
        for p in cliques_prop:
            pharmacophore[p] = 1
        
        return np.array(pharmacophore)

    def getCliqueEdgeFeatures_pharmacophore(self, edge, clique, edge_idx, cliques_prop, mol):
        # number of edge features (3)
        begin = cliques_prop[edge[0]]
        end = cliques_prop[edge[1]]
        
        if len(begin) == 0:
            begin_type = 'none'
        else:
            begin_type = 'phar'
            
        if len(end) == 0:
            end_type = 'none'
        else:
            end_type = 'phar'
        
        definedEdgeType = {('none', 'none'): 0,
                           ('none', 'phar'): 1,
                           ('phar', 'none'): 1,
                           ('phar', 'phar'): 2,}
        EdgeType = definedEdgeType[(begin_type, end_type)]
        
        return np.array(one_of_k_encoding(EdgeType,list(set(definedEdgeType.values()))))


    def mol_to_graph(self, mol, cliques, edges, cliques_prop, normalize=True):
        clique_size = len(cliques)
        
        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            clique_features = self.getCliqueFeatures_pharmacophore(cliques[idx], edges, idx, cliques_prop[idx], mol)
            if normalize and sum(clique_features) != 0:
                clique_attr.append(list(clique_features)/sum(clique_features))
            else:
                clique_attr.append(clique_features)
        
        cliqueedge_index = edges
        cliqueedge_attr = []
        for idx in range(len(edges)):
            cliqueedge_features = self.getCliqueEdgeFeatures_pharmacophore(edges[idx], cliques, idx, cliques_prop, mol)
            if normalize and sum(cliqueedge_features) != 0:
                cliqueedge_attr.append(list(cliqueedge_features)/sum(cliqueedge_features))
            else:
                cliqueedge_attr.append(cliqueedge_features)

        return clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr


# SUBSTRUCTURE-based
class SubstructureGraph(MoleculeGraph):
    def __init__(self, smiles, tokenizer, normalize=True):
        self.smiles = mol_to_smiles(smiles_to_mol(smiles, with_atom_index=False))
        self.mol = smiles_to_mol(self.smiles, with_atom_index=False)
        self.tokenizer = tokenizer
        self.normalize = normalize
        self.cliques_smiles = dict()
        cliques, edges, cliques_attr, edges_attr = self.getReducedGraph(self.mol, tokenizer)
        self.cliques = cliques
        self.edges = edges
        self.cliques_attr = cliques_attr
        self.edges_attr = edges_attr
        clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_graph(self.mol, cliques, edges, cliques_attr, edges_attr, normalize)
        self.graph_size = clique_size
        self.node_index = clique_index
        self.node_features = clique_attr
        self.edge_index = cliqueedge_index
        self.edge_features = cliqueedge_attr

    def getReducedGraph(self, mol, tokenizer):
        smiles = mol_to_smiles(mol)
        # print('Process smiles', smiles)
        mol_nx, cliques, edges, cliques_attr, edges_attr = tokenizer.tokenize(smiles)
        # print('Tokenized Molecule NX:')
        # print(mol_nx)
        # print('Input smiles VS Reconstructed smiles')
        # print(smiles, 'VS', mol_nx.to_smiles())
        return cliques, edges, cliques_attr, edges_attr

    def generate_morgan(self, mol, radius=2, nBits=1024):
        morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        return morgan

    def getCliqueFeatures_substructure(self, clique, edges, clique_idx, cliques_attr, edges_attr, mol):
        # number of node features (1024)
        substructure = cliques_attr[clique_idx]
        substructure_mol = smiles_to_mol(substructure, with_atom_index=False, kekulize=True)
        self.cliques_smiles[clique_idx] = [mol_to_smiles(substructure_mol)]
        morgan = self.generate_morgan(substructure_mol)
        return np.array(morgan)

    def getCliqueEdgeFeatures_substructure(self, edge, clique, edge_idx, cliques_attr, edges_attr, mol): 
        # number of edge features (14) # (16)
        connection = len(edges_attr[edge_idx])
        if connection == 1:
            edge_a = edges_attr[edge_idx][0]
            begin_atom = edge_a[0]+1
            end_atom = edge_a[1]+1
            bond_type = edge_a[2]
        else:
            begin_atom = 0
            end_atom = 0
            bond_type = None
        # return np.array([begin_atom]+[end_atom]+
        #                 one_of_k_encoding_unk(connection,list(range(10)))+
        #                 one_of_k_encoding_none(bond_type, definedBond))
        return np.array(one_of_k_encoding_unk(connection,list(range(10)))+
                        one_of_k_encoding_none(bond_type, definedBond))

    def mol_to_graph(self, mol, cliques, edges, cliques_attr, edges_attr, normalize=True):
        clique_size = len(cliques)

        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            clique_features = self.getCliqueFeatures_substructure(cliques[idx], edges, idx, cliques_attr, edges_attr, mol)
            if normalize and sum(clique_features) != 0:
                clique_attr.append(list(clique_features)/sum(clique_features))
            else:
                clique_attr.append(clique_features)
        
        cliqueedge_index = edges
        cliqueedge_attr = []
        for idx in range(len(edges)):
            cliqueedge_features = self.getCliqueEdgeFeatures_substructure(edges[idx], cliques, idx, cliques_attr, edges_attr, mol)
            if normalize and sum(cliqueedge_features) != 0:
                cliqueedge_attr.append(list(cliqueedge_features)/sum(cliqueedge_features))
            else:
                cliqueedge_attr.append(cliqueedge_features)
        
        return clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr
