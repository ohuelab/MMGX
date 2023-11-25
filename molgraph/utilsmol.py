######################
### Import Library ###
######################

from rdkit import Chem
import networkx as nx

##########################
### Utilities Function ###
##########################

# smiles to mol
def mol_to_smiles(mol):
    smiles = Chem.MolToSmiles(mol)
    return smiles

# mol to smiles
def smiles_to_mol(smiles, with_atom_index=True, kekulize=False):
    mol = Chem.MolFromSmiles(smiles)
    if with_atom_index:
        mol = mol_with_atom_index(mol)
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol

# mol with atom index
def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

# check and get valid smiles
def getValidSmiles(smiles):
    smi = mol_to_smiles(smiles_to_mol(smiles, with_atom_index=False))
    return smi

# get whether smiles is single atom
def checkSingleAtom(smiles):
    return smiles_to_mol(smiles, with_atom_index=False).GetNumAtoms()==1

# convert mol to topology (atom symbol, no consider bond type)
def topology_checker(mol):
    topology = nx.Graph()
    for atom in mol.GetAtoms():
        # Add the atoms as nodes
        topology.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
    for bond in mol.GetBonds():
        # Add the bonds as edges
        topology.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), type=bond.GetBondType())
        # # Add the bonds as edges
        # for bonded in atom.GetNeighbors():
        #     topology.add_edge(atom.GetIdx(), bonded.GetIdx())
    return topology

# check same atom (atom from nx)
def is_same_atom(a1, a2):
    return a1['symbol'] == a2['symbol']

# check same bond (bond from nx)
def is_same_bond(b1, b2):
    return b1['type'] == b2['type']

# check graph is isomorphic
def is_isomorphic(topology1, topology2):
    return nx.is_isomorphic(topology1, topology2, node_match=is_same_atom, edge_match=is_same_bond)

# check graph is isomorphic only atom
def is_isomorphic_atom(topology1, topology2):
    return nx.is_isomorphic(topology1, topology2, node_match=is_same_atom)