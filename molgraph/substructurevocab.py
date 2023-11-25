######################
### Import Library ###
######################

# my library
from molgraph.utilsonehot import *
from molgraph.utilsmol import *
from molgraph.utilsgraph import *
# standard
import numpy as np
import copy as copy
from tqdm import tqdm
# rdkit
from rdkit import Chem
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
# graph
import networkx as nx

#########################
### Global Definition ###
#########################

MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':5, 'O':2, 'P':5, 'S':6} #, 'Se':4, 'Si':4}
Bond_List = [None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

##########################
### Utilities Function ###
##########################

# get mol from bond idx from atom idx
def get_submol(mol, atom_indices):
    if len(atom_indices) == 1:
        atom_smiles = mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()
        mol = smiles_to_mol(atom_smiles, with_atom_index=False, kekulize=True)
        return mol
    aid_dict = { i: True for i in atom_indices }
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        begin_aid = bond.GetBeginAtomIdx()
        end_aid = bond.GetEndAtomIdx()
        if begin_aid in aid_dict and end_aid in aid_dict:
            edge_indices.append(i)
    mol = Chem.PathToSubmol(mol, edge_indices)
    return mol

# get smiles from atom idx
def get_subsmiles(mol, atom_indices):
    # if len(atom_indices) == 1:
    #     atom_smiles = mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()
    #     # single H atom
    #     if atom_smiles == 'H' or atom_smiles == 'Na':
    #         atom_smiles = Chem.MolFragmentToSmiles(mol, atom_indices)
    #     return atom_smiles
    smiles = Chem.MolFragmentToSmiles(mol, atom_indices)
    return smiles

# get dict of mol atom idx mapping to submol atom idx
def get_submol_atom_map(mol, submol, group):
    # get smiles from submol
    smi = mol_to_smiles(submol)
    submol = smiles_to_mol(smi, with_atom_index=False, kekulize=True)
    # special with N+ and N-
    for atom in submol.GetAtoms():
        if atom.GetSymbol() == 'N' and (atom.GetExplicitValence() == 3 and atom.GetFormalCharge() == 1) or atom.GetExplicitValence() < 3:
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(2)
        elif atom.GetSymbol() == 'As':
            atom.SetNumRadicalElectrons(0)
    # get substructure atom idx of submol in mol
    matches = mol.GetSubstructMatches(submol)
    # check matches
    if len(matches) == 0:
        for atom in submol.GetAtoms():
            atom.SetNumRadicalElectrons(0)
        matches = mol.GetSubstructMatches(submol)
    if len(matches) == 0: # maybe fused ring (need to set unspecified bond type)
        for b in submol.GetBonds():
            # print(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
            b.SetBondType(Chem.rdchem.BondType.UNSPECIFIED)
        matches = mol.GetSubstructMatches(submol)
    # print('matches', len(matches))
    if len(matches) >= 500:
        matches = mol.GetSubstructMatches(submol, maxMatches=3000)
    if len(matches) == 0:
        print('NOT MATCH ERROR', smi, group, mol_to_smiles(mol))
    # old atom idx to new atom idx
    old2new = { i: 0 for i in group }  
    found = False
    for m in matches:
        hit = True
        for i, atom_idx in enumerate(m):
            if atom_idx not in old2new:
                hit = False
                break
            old2new[atom_idx] = i
        if hit:
            found = True
            break
    assert found
    return old2new

# count number of each (predefined) atom in molinpiece
def count_atom(molinp, return_dict=False):
    atom_dict = { atom: 0 for atom in MAX_VALENCE }
    for atom in molinp.mol.GetAtoms():
        if atom.GetSymbol() in atom_dict:
            atom_dict[atom.GetSymbol()] += 1
    if return_dict:
        return atom_dict
    else:
        return sum(atom_dict.values())

# count number of subsmiles in molinpiece
def count_freq(molinp):
    freqs = {}
    nei_smis = molinp.get_nei_smis()
    for smi in nei_smis:
        freqs.setdefault(smi, 0)
        freqs[smi] += 1
    return freqs, molinp

############################
### Class Initialization ###
############################

class MolInPiece:
    def __init__(self, mol):
        self.mol = mol
        self.smi = mol_to_smiles(mol)
         # pid is the key (init by all atom idx)
        self.pieces, self.pieces_smis = {}, {} 
        for atom in mol.GetAtoms():
            idx, symbol = atom.GetIdx(), atom.GetSymbol()
            self.pieces[idx] = { idx: symbol }
            self.pieces_smis[idx] = symbol
        # assign atom idx to pid
        self.inversed_index = {} 
        self.upid_cnt = len(self.pieces)
        for aid in range(mol.GetNumAtoms()):
            for key in self.pieces:
                piece = self.pieces[key]
                if aid in piece:
                    self.inversed_index[aid] = key
        self.dirty = True
        # not public, record neighboring graphs and their pids
        self.smi2pids = {} 

    def get_nei_pieces(self):
        nei_pieces, merge_pids = [], []
        for key in self.pieces:
            piece = self.pieces[key]
            local_nei_pid = []
            for aid in piece:
                atom = self.mol.GetAtomWithIdx(aid)
                for nei in atom.GetNeighbors():
                    nei_idx = nei.GetIdx()
                    # only consider connecting to former atoms
                    if nei_idx in piece or nei_idx > aid:   
                        continue
                    # get pid from atom id
                    local_nei_pid.append(self.inversed_index[nei_idx]) 
            local_nei_pid = set(local_nei_pid) # unique
            for nei_pid in local_nei_pid: # loop nei pid
                new_piece = copy.copy(piece)
                new_piece.update(self.pieces[nei_pid]) # update nei (add new nei)
                nei_pieces.append(new_piece) # add new dict of key and nei
                merge_pids.append((key, nei_pid)) # add new tuple of pid and set of nei pid
        return nei_pieces, merge_pids

    def get_nei_smis(self):
        if self.dirty:
            nei_pieces, merge_pids = self.get_nei_pieces()
            nei_smis, self.smi2pids = [], {}
            for i, piece in enumerate(nei_pieces):
                smi = get_subsmiles(self.mol, list(piece.keys()))
                nei_smis.append(smi)
                self.smi2pids.setdefault(smi, [])
                self.smi2pids[smi].append(merge_pids[i])
            self.dirty = False
        else:
            nei_smis = list(self.smi2pids.keys())
        return nei_smis

    def merge(self, smi):
        if self.dirty:
            self.get_nei_smis()
        if smi in self.smi2pids:
            merge_pids = self.smi2pids[smi]
            for pid1, pid2 in merge_pids:
                if pid1 in self.pieces and pid2 in self.pieces: # possibly del by former
                    self.pieces[pid1].update(self.pieces[pid2])
                    self.pieces[self.upid_cnt] = self.pieces[pid1]
                    self.pieces_smis[self.upid_cnt] = smi
                    for aid in self.pieces[pid2]:
                        self.inversed_index[aid] = pid1
                    for aid in self.pieces[pid1]:
                        self.inversed_index[aid] = self.upid_cnt
                    del self.pieces[pid1]
                    del self.pieces[pid2]
                    del self.pieces_smis[pid1]
                    del self.pieces_smis[pid2]
                    self.upid_cnt += 1
        self.dirty = True   # revised

    def get_smis_pieces(self):
        # return list of tuple(smi, idxs)
        res = []
        for pid in self.pieces_smis:
            smi = self.pieces_smis[pid]
            group_dict = self.pieces[pid]
            idxs = list(group_dict.keys())
            res.append((smi, idxs))
        return res

#######################
### Vocab Generator ###
#######################

def generate_vocab(smiles_list, vocab_len, vocab_path):
    # load molecules
    smis = [mol_to_smiles(smiles_to_mol(s, with_atom_index=False, kekulize=True)) for s in smiles_list]
    # init to atoms
    mols = [MolInPiece(smiles_to_mol(smi, with_atom_index=False, kekulize=True)) for smi in smis]
    
    # details: {smi: [count_atom, frequency]}
    details = dict()
    # subsmiles list (initialize with predefined atom)
    selected_smis = list(MAX_VALENCE.keys())
    
    # calculate single atom frequency
    for atom in selected_smis:
        # frequency of single atom is not calculated
        details[atom] = [1, 0]  
    for mol in mols:
        cnts = count_atom(mol, return_dict=True)
        for atom in details:
            if atom in cnts:
                details[atom][1] += cnts[atom]
    
    # generate_vocab process
    add_len = vocab_len - len(selected_smis)
    for _ in tqdm(range(add_len), desc='===Generating vocab smiles'):
        res_list = [count_freq(m) for m in mols]
        freqs, mols = {}, []
        for freq, mol in res_list:
            mols.append(mol)
            for key in freq:
                freqs.setdefault(key, 0)
                freqs[key] += freq[key]
        # find the piece to merge
        max_count, merge_smi = 0, ''
        for smi in freqs:
            count = freqs[smi]
            if count > max_count:
                max_count = count
                merge_smi = smi
        # merge
        for mol in mols:
            mol.merge(merge_smi)
        selected_smis.append(merge_smi)
        merge_smi_molinp = MolInPiece(smiles_to_mol(merge_smi, with_atom_index=False, kekulize=True))
        details[merge_smi] = [count_atom(merge_smi_molinp), max_count]

    # sorting vocab by atom num
    selected_smis.sort(key=lambda x: details[x][0], reverse=True)
    with open(vocab_path, 'w') as fout:
        fout.writelines(list(map(lambda smi: f'{smi}\t{details[smi][0]}\t{details[smi][1]}\n', selected_smis)))
    
    return


#################
### Tokenizer ###
#################

class AtomVocab:
    def __init__(self, atom_special=None, bond_special=None):
        # descriptor
        self.max_Hs = max(MAX_VALENCE.values())
        self.min_formal_charge = -2
        self.max_formal_charge = 2
        # atom
        self.idx2atom = list(MAX_VALENCE.keys())
        if atom_special is None:
            atom_special = []
        self._atom_pad = '<apad>'
        atom_special.append(self._atom_pad)
        self.idx2atom += atom_special
        self.atom2idx = { atom: i for i, atom in enumerate(self.idx2atom) }
        # bond
        self.idx2bond = copy.copy(Bond_List)
        self._bond_pad = '<bpad>'
        if bond_special is None:
            bond_special = []
        bond_special.append(self._bond_pad)
        self.idx2bond += bond_special
        self.bond2idx = { bond: i for i, bond in enumerate(self.idx2bond) }
        # special atom and bond
        self.atom_special = atom_special
        self.bond_special = bond_special

    def idx_to_atom(self, idx):
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        return self.atom2idx[atom]

    def atom_pad(self):
        return self._atom_pad

    def atom_pad_idx(self):
        return self.atom_to_idx(self.atom_pad())

    def idx_to_bond(self, idx):
        return self.idx2bond[idx]

    def bond_to_idx(self, bond):
        return self.bond2idx[bond]

    def bond_pad(self):
        return self._bond_pad

    def bond_pad_idx(self):
        return self.bond_to_idx(self.bond_pad())

    def bond_idx_to_valence(self, idx):
        bond_enum = self.idx2bond[idx]
        if bond_enum == Chem.BondType.SINGLE:
            return 1
        elif bond_enum == Chem.BondType.DOUBLE:
            return 2
        elif bond_enum == Chem.BondType.TRIPLE:
            return 3
        else:   
            return -1 # invalid bond

    def num_atom_type(self):
        return len(self.idx2atom)

    def num_bond_type(self):
        return len(self.idx2bond)

class PieceNode:
    def __init__(self, smiles: str, pos: int, group: list, atom_mapping: dict):
        self.smiles = smiles
        self.pos = pos
        self.group = group
        self.mol = smiles_to_mol(smiles, with_atom_index=False, kekulize=True)
        # map atom idx in the whole mol to those in the piece
        self.atom_mapping = copy.copy(atom_mapping)  

    def get_mol(self):
        '''return molecule in rdkit form'''
        return self.mol

    def get_atom_mapping(self):
        return copy(self.atom_mapping)

    def __str__(self):
        return f'''
                    smiles: {self.smiles},
                    position: {self.pos}
                '''

class PieceEdge:
    def __init__(self, src: int, dst: int, edges: list):
        # list of tuple (a, b, type) where the canonical order is used
        self.edges = copy.copy(edges)  
        self.src = src
        self.dst = dst
        self.dummy = False
        if len(self.edges) == 0:
            self.dummy = True

    def get_edges(self):
        return copy(self.edges)

    def get_num_edges(self):
        return len(self.edges)

    def __str__(self):
        return f'''
                    src piece: {self.src}, dst piece: {self.dst},
                    atom bonds: {self.edges}
                '''

class Molecule(nx.Graph):
    '''molecule represented in piece-level'''

    def __init__(self, smiles: str=None, groups: list=None):
        super().__init__()
        if smiles is None:
            return
        self.graph['smiles'] = smiles
        rdkit_mol = smiles_to_mol(smiles, with_atom_index=False, kekulize=True)
        # processing atoms
        aid2pos = {}
        for pos, group in enumerate(groups):
            for aid in group:
                aid2pos[aid] = pos
            piece_smi = get_subsmiles(rdkit_mol, group)
            piece_mol = smiles_to_mol(piece_smi, with_atom_index=False, kekulize=True)
            atom_mapping = get_submol_atom_map(rdkit_mol, piece_mol, group)
            node = PieceNode(piece_smi, pos, group, atom_mapping)
            self.add_node(node)
        # process edges
        edges_arr = [[[] for _ in groups] for _ in groups]  # adjacent
        for edge_idx in range(rdkit_mol.GetNumBonds()):
            bond = rdkit_mol.GetBondWithIdx(edge_idx)
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            begin_piece_pos = aid2pos[begin]
            end_piece_pos = aid2pos[end]
            begin_mapped = self.nodes[begin_piece_pos]['piece'].atom_mapping[begin]
            end_mapped = self.nodes[end_piece_pos]['piece'].atom_mapping[end]
            bond_type = bond.GetBondType()
            edges_arr[begin_piece_pos][end_piece_pos].append((begin_mapped, end_mapped, bond_type))
            edges_arr[end_piece_pos][begin_piece_pos].append((end_mapped, begin_mapped, bond_type))
        # add egdes into the graph
        self.edge_record = dict()
        for i in range(len(groups)):
            for j in range(len(groups)):
                # if not i < j or len(edges_arr[i][j]) == 0: # i->j and j->i
                if i == j or len(edges_arr[i][j]) == 0:
                    continue
                edge = PieceEdge(i, j, edges_arr[i][j])
                self.edge_record[(i, j)] = edges_arr[i][j]
                self.add_edge(edge)

    @classmethod
    def from_nx_graph(cls, graph: nx.Graph, deepcopy=True):
        if deepcopy:
            graph = deepcopy(graph)
        graph.__class__ = Molecule
        return graph

    @classmethod
    def merge(cls, mol0, mol1, edge=None):
        # reorder
        node_mappings = [{}, {}]
        mols = [mol0, mol1]
        mol = Molecule.from_nx_graph(nx.Graph())
        for i in range(2):
            for n in mols[i].nodes:
                node_mappings[i][n] = len(node_mappings[i])
                node = copy.deepcopy(mols[i].get_node(n))
                node.pos = node_mappings[i][n]
                mol.add_node(node)
            for src, dst in mols[i].edges:
                edge = copy.deepcopy(mols[i].get_edge(src, dst))
                edge.src = node_mappings[i][src]
                edge.dst = node_mappings[i][dst]
                mol.add_edge(src, dst, connects=edge)
        # add new edge
        edge = copy.deepcopy(edge)
        edge.src = node_mappings[0][edge.src]
        edge.dst = node_mappings[1][edge.dst]
        mol.add_edge(edge)
        return mol

    def get_edge(self, i, j) -> PieceEdge:
        return self[i][j]['connects']

    def get_node(self, i) -> PieceNode:
        return self.nodes[i]['piece']

    def add_edge(self, edge: PieceEdge):
        src, dst = edge.src, edge.dst
        super().add_edge(src, dst, connects=edge)

    def add_node(self, node: PieceNode):
        n = node.pos
        super().add_node(n, piece=node)

    def subgraph(self, nodes: list, deepcopy=True):
        graph = super().subgraph(nodes)
        assert isinstance(graph, Molecule)
        return graph

    def to_rdkit_mol(self):
        mol = Chem.RWMol()
        aid_mapping = {}
        # add all the pieces to rwmol
        for n in self.nodes:
            piece = self.get_node(n)
            submol = piece.get_mol()
            for atom in submol.GetAtoms():
                new_atom = Chem.Atom(atom.GetSymbol())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                mol.AddAtom(atom)
                aid_mapping[(n, atom.GetIdx())] = len(aid_mapping)
            for bond in submol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                begin, end = aid_mapping[(n, begin)], aid_mapping[(n, end)]
                mol.AddBond(begin, end, bond.GetBondType())
        for src, dst in self.edges:
            piece_edge = self.get_edge(src, dst)
            pid_src, pid_dst = piece_edge.src, piece_edge.dst
            for begin, end, bond_type in piece_edge.edges:
                begin, end = aid_mapping[(pid_src, begin)], aid_mapping[(pid_dst, end)]
                mol.AddBond(begin, end, bond_type)
        mol = mol.GetMol()
        # sanitize, firstly handle mal-formed N+
        mol.UpdatePropertyCache(strict=False)
        ps = Chem.DetectChemistryProblems(mol)
        if not ps:
            Chem.SanitizeMol(mol)
            return mol
        for p in ps:
            if p.GetType()=='AtomValenceException':  # N
                at = mol.GetAtomWithIdx(p.GetAtomIdx())
                if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                    at.SetFormalCharge(1)
        Chem.SanitizeMol(mol)
        return mol

    def to_smiles(self):
        rdkit_mol = self.to_rdkit_mol()
        return mol_to_smiles(rdkit_mol)

    def __str__(self):
        desc = 'nodes: \n'
        for ni, node in enumerate(self.nodes):
            desc += f'{ni}:{self.get_node(node)}\n'
        desc += 'edges: \n'
        for src, dst in self.edges:
            desc += f'{src}-{dst}:{self.get_edge(src, dst)}\n'
        return desc


class Tokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        self.vocab_dict = {}
        self.idx2piece, self.piece2idx = [], {}
        self.max_num_nodes = 0
        for line in lines:
            smi, atom_num, freq = line.strip().split('\t')
            self.vocab_dict[smi] = (int(atom_num), int(freq))
            self.max_num_nodes = max(self.max_num_nodes, int(atom_num))
            self.piece2idx[smi] = len(self.idx2piece)
            self.idx2piece.append(smi)
        self.pad, self.end = '<pad>', '<s>'
        for smi in [self.pad, self.end]:
            self.piece2idx[smi] = len(self.idx2piece)
            self.idx2piece.append(smi)
        # for fine-grained level (atom level)
        self.bond_start = '<bstart>'
        self.atom_level_vocab = AtomVocab(bond_special=[self.bond_start])
        self.max_num_nodes += 2 # start, padding

    def tokenize(self, mol):
        cliques = []
        smiles = mol
        if isinstance(mol, str):
            mol = smiles_to_mol(mol, with_atom_index=False, kekulize=True)
        else:
            smiles = mol_to_smiles(mol)
        rdkit_mol = mol
        mol = MolInPiece(mol)
        while True:
            nei_smis = mol.get_nei_smis()
            max_freq, merge_smi = -1, ''
            for smi in nei_smis:
                if smi not in self.vocab_dict:
                    continue
                freq = self.vocab_dict[smi][1]
                if freq > max_freq:
                    max_freq, merge_smi = freq, smi
            if max_freq == -1:
                break
            mol.merge(merge_smi)
        res = mol.get_smis_pieces()
        # construct reversed index
        aid2pid = {}
        for pid, piece in enumerate(res):
            _, aids = piece
            for aid in aids:
                aid2pid[aid] = pid
        # construct adjacent matrix
        ad_mat = [[0 for _ in res] for _ in res]
        for aid in range(rdkit_mol.GetNumAtoms()):
            atom = rdkit_mol.GetAtomWithIdx(aid)
            for nei in atom.GetNeighbors():
                nei_id = nei.GetIdx()
                i, j = aid2pid[aid], aid2pid[nei_id]
                if i != j:
                    ad_mat[i][j] = ad_mat[j][i] = 1
        # np.random.shuffle(res)
        group_idxs = [x[1] for x in res]
        new_molecule = Molecule(smiles, group_idxs)
        cliques = []
        cliques_attr = []
        for n in new_molecule.nodes():
            cliques.append(new_molecule.get_node(n).group)
            cliques_attr.append(new_molecule.get_node(n).smiles)
        edges = []
        edges_attr = []
        for e in new_molecule.edges():
            edges.append((e[0], e[1]))
            edges.append((e[1], e[0]))
            edges_attr.append(new_molecule.edge_record[e[0], e[1]])
            edges_attr.append(new_molecule.edge_record[e[1], e[0]])
        return new_molecule, cliques, edges, cliques_attr, edges_attr

    def idx_to_piece(self, idx):
        return self.idx2piece[idx]

    def piece_to_idx(self, piece):
        return self.piece2idx[piece]

    def pad_idx(self):
        return self.piece2idx[self.pad]

    def end_idx(self):
        return self.piece2idx[self.end]

    def atom_vocab(self):
        return copy(self.atom_level_vocab)

    def num_piece_type(self):
        return len(self.idx2piece)

    def atom_pos_pad_idx(self):
        return self.max_num_nodes - 1

    def atom_pos_start_idx(self):
        return self.max_num_nodes - 2

    def __call__(self, mol):
        return self.tokenize(mol)

    def __len__(self):
        return len(self.idx2piece)