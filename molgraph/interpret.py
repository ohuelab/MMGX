######################
### Import Library ###
######################

# My library
from molgraph.dataset import *
from molgraph.molgraph import *
from molgraph.utilsmol import *
from molgraph.utilsgraph import *

# general
import os as os
import numpy as np
import random as random
import copy as copy

# scikit scipy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.special import softmax

# visualize
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage.io import imread
from cairosvg import svg2png, svg2ps

# rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
rdDepictor.SetPreferCoordGen(True)

# graph
import networkx as nx

##########################
### Interpret Function ###
##########################

def display_interpret_graph(mol, mask_graph, scale=False, colors=None):
    
    mask_graph_atom_key = list(mask_graph['atom'].keys())
    mask_graph_atom_value = list(mask_graph['atom'].values())
    mask_graph_bond_key = list(mask_graph['bond'].keys())
    mask_graph_bond_value = list(mask_graph['bond'].values())
    
    mask_graph_color = copy.deepcopy(mask_graph)
    
    if scale:
        for i in mask_graph_color['atom']:
            abs_color = mask_graph_color['atom'][i]
            mask_graph_color['atom'][i] = colors.to_rgba(abs_color)
        for i in mask_graph_color['bond']:
            abs_color = mask_graph_color['bond'][i]
            mask_graph_color['bond'][i] = colors.to_rgba(abs_color)
    else:
        colors = colors
        for i in mask_graph_color['atom']:
            color = colors[mask_graph_color['atom'][i]]
            mask_graph_color['atom'][i] = tuple(color)
        for i in mask_graph_color['bond']:
            color = colors[mask_graph_color['bond'][i]]
            mask_graph_color['bond'][i] = tuple(color)
    
    rdDepictor.Compute2DCoords(mol)
    dg = rdMolDraw2D.MolDraw2DSVG(1050, 1000)
    dg.SetFontSize(1)
    
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    dg.DrawMolecule(mol, highlightAtoms=mask_graph_atom_key,
                    highlightAtomColors=mask_graph_color['atom'],
                    highlightBonds=mask_graph_bond_key,
                    highlightBondColors=mask_graph_color['bond'])
    
    dg.FinishDrawing()
    svg = dg.GetDrawingText()
    svg = svg.replace('svg:', '')
    svg2png(bytestring=svg, write_to='tmp_graph_inp.png', dpi=300)
    img = imread('tmp_graph_inp.png')
    os.remove('tmp_graph_inp.png')
    return img


def display_interpret_reduced(mol, mask_graph, edges, mol_pos=None, scale=False, colors=None):
    
    mask_graph_atom_key = list(mask_graph['atom'].keys())
    mask_graph_atom_value = list(mask_graph['atom'].values())
    mask_graph_bond_key = list(mask_graph['bond'].keys())
    mask_graph_bond_value = list(mask_graph['bond'].values())
    
    mask_graph_color = copy.deepcopy(mask_graph)
    
    if scale:
        for i in mask_graph_color['atom']:
            abs_color = mask_graph_color['atom'][i]
            mask_graph_color['atom'][i] = colors.to_rgba(abs_color)
        for i in mask_graph_color['bond']:
            abs_color = mask_graph_color['bond'][i]
            mask_graph_color['bond'][i] = colors.to_rgba(abs_color)
            
        colors = list(mask_graph_color['atom'].values())
    else:
        colors = colors
        
    mol_nx = g_to_nx(mask_graph_atom_key, edges)
    mol_pos = mol_pos
    mol_node = nx.get_node_attributes(mol_nx, 'idx')
    mol_colors = colors

    fig = plt.figure(figsize=(10,10))
    plt.clf()
    nx.draw(mol_nx, 
            pos=mol_pos,
            labels=mol_node,
            with_labels=True,
            node_color=mol_colors,
            node_size=3000,
            font_size='20')
            
    plt.savefig('tmp_reduced_inp.png', dpi=300)
    plt.close(fig)
    img = plt.imread('tmp_reduced_inp.png')
    os.remove('tmp_reduced_inp.png')
    return img

# def display_interpret(mol, cliques, edges):
#     # formatting mol
#     mol = mol_with_atom_index(mol)
#     AllChem.Compute2DCoords(mol)
#     number_of_sub = len(cliques)
    
#     # random colors
#     colors = []
#     for i in range(number_of_sub):
#         r = random.randint(0,255)/255
#         g = random.randint(0,255)/255
#         b = random.randint(0,255)/255
#         colors.append([r,g,b,1])
    
#     # construct mask graph
#     mask_graph = {}
#     mask_graph['atom'] = {}
#     mask_graph['bond'] = {}
#     for i, c in enumerate(cliques):
#         for a in c:
#             # should have multiple
#             # mask_graph['atom'].append({a: i})
#             mask_graph['atom'][a] = i
#             for j, b in enumerate(mol.GetBonds()):
#                 if all(bb in c for bb in [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]):
#                     mask_graph['bond'][j] =  i
                    
#     # construct mast reduced
#     mask_reduced = {}
#     mask_reduced['atom'] = {}
#     mask_reduced['bond'] = {}
#     for i, e in enumerate(cliques):
#         mask_reduced['atom'][i] = i
#         mask_reduced['atom'][i] = i

#     # visualize
#     fig, ax = plt.subplots(1, 2, figsize=(15,7))
    
#     # mol with sub
#     ax1 = plt.subplot(1, 2, 1)
    
#     ax1.grid(False)
#     ax1.axis('off')
#     ax1.imshow(display_interpret_graph(mol, mask_graph, scale=False, colors=colors))
    
#     # nx mol
#     ax2 = plt.subplot(1, 2, 2)
    
#     # compute position
#     mol_con = [c.GetPositions()[:,:2] for c in mol.GetConformers()][0]
#     mol_pos_node = {i: c for i, c in enumerate(mol_con)}
#     mol_pos = dict()
#     for i_c, c in enumerate(cliques):
#         b_pos_x = 0
#         b_pos_y = 0
#         b_num = 0
#         for n in c:
#             b_pos_x += mol_pos_node[n][0]
#             b_pos_y += mol_pos_node[n][1]
#             b_num += 1
#         if b_num != 0:
#             mol_pos[i_c] = [b_pos_x/b_num, b_pos_y/b_num]
    
#     display_interpret_reduced(mol, mask_reduced, edges, mol_pos=mol_pos, scale=False, colors=colors)


def display_interpret_weight(mol, cliques, edges, mask_graph_g, mask_graph_r, scale=True, color_map='Greens'):
    # formatting mol
    mol = mol_with_atom_index(mol)
    AllChem.Compute2DCoords(mol)
    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap(color_map)
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    fig, ax = plt.subplots(1, 3, figsize=(30,10))
    plt.axis('off')

    if mask_graph_g is not None:

        inp_graph = display_interpret_graph(mol, mask_graph_g, scale=scale, colors=plt_colors)
        
        # mol with sub
        ax1 = plt.subplot(1, 3, 1)
        ax1.grid(False)
        ax1.axis('off')
        ax1.set_axis_off()
        ax1.imshow(inp_graph)

    if mask_graph_r is not None:
        
        # compute position
        mol_con = [c.GetPositions()[:,:2] for c in mol.GetConformers()][0]
        mol_pos_node = {i: c for i, c in enumerate(mol_con)}
        mol_pos = dict()
        for i_c, c in enumerate(cliques):
            b_pos_x = 0
            b_pos_y = 0
            b_num = 0
            for n in c:
                b_pos_x += mol_pos_node[n][0]
                b_pos_y += mol_pos_node[n][1]
                b_num += 1
            if b_num != 0:
                mol_pos[i_c] = [b_pos_x/b_num, b_pos_y/b_num]

        inp_reduced = display_interpret_reduced(mol, mask_graph_r, edges, mol_pos=mol_pos, scale=scale, colors=plt_colors)

        # mol-sub with sub
        ax2 = plt.subplot(1, 3, 2)
        ax2.grid(False)
        ax2.axis('off')
        ax2.set_axis_off()
        ax2.imshow(inp_reduced)
        
        # construct mask graph
        mask_graph_rtog = {}
        mask_graph_rtog['atom'] = {}
        mask_graph_rtog['bond'] = {}
        
        for i, n in enumerate(range(mol.GetNumAtoms())):
            mask_graph_rtog['atom'][n] = 0

        for i, c in enumerate(cliques):
            for a in c:
                # overlap node between cliques
                mask_graph_rtog['atom'][a] += mask_graph_r['atom'][i] # sum
                # mask_graph_rtog['atom'][a] = max(mask_graph_rtog['atom'][a], mask_graph_r['atom'][i]) # max

        mask_graph_rtog = minmaxnormalize(mask_graph_rtog)

        inp_reduced_graph = display_interpret_graph(mol, mask_graph_rtog, scale=scale, colors=plt_colors)

        # mol-sub with sub
        ax3 = plt.subplot(1, 3, 3)
        ax3.grid(False)
        ax3.axis('off')
        ax3.set_axis_off()
        ax3.imshow(inp_reduced_graph)

    return fig


def mask_graph(d_att_g):
    mask_graph_g = {'atom': dict(), 'bond': dict()}

    d_att_g_index = np.array(d_att_g[0], dtype=int)
    d_att_g_weight = np.array(d_att_g[1], dtype=float)

    for i, w in zip(d_att_g_index[0], d_att_g_weight):
        mask_graph_g['atom'][int(i)] = float(w.item(0))

    mask_graph_g = minmaxnormalize(mask_graph_g)

    return mask_graph_g

def mask_reduced(d_att_r):
    mask_graph_r = {'atom': dict(), 'bond': dict()}

    d_att_r_index = np.array(d_att_r[0])
    d_att_r_weight = np.array(d_att_r[1])

    for i, w in zip(d_att_r_index[0], d_att_r_weight):
        mask_graph_r['atom'][int(i)] = float(w.item(0))

    mask_graph_r = minmaxnormalize(mask_graph_r)

    return mask_graph_r


def mask_rtog(smiles, cliques, mask_graph_r):
    # construct mask graph
    mask_graph_rtog = {}
    mask_graph_rtog['atom'] = {}
    mask_graph_rtog['bond'] = {}

    mol = smiles_to_mol(smiles, with_atom_index=True)
    for i, n in enumerate(range(mol.GetNumAtoms())):
        mask_graph_rtog['atom'][n] = 0

    for i, c in enumerate(cliques):
        for a in c:
            # overlap node between cliques
            mask_graph_rtog['atom'][a] += mask_graph_r['atom'][i] # sum
            # mask_graph_rtog['atom'][a] = max(mask_graph_rtog['atom'][a], mask_graph_r['atom'][i]) # max

    mask_graph_rtog = minmaxnormalize(mask_graph_rtog)

    return mask_graph_rtog


def mask_gandr(d_att_g, mask_graph_rtog):
    # construct mask graph
    mask_graph_gandr = {}
    mask_graph_gandr['atom'] = {}
    mask_graph_gandr['bond'] = {}

    for g, rtog in zip(d_att_g['atom'], mask_graph_rtog['atom']):
        mask_graph_gandr['atom'][g] = max([d_att_g['atom'][g], mask_graph_rtog['atom'][rtog]])

    mask_graph_gandr = minmaxnormalize(mask_graph_gandr)

    return mask_graph_gandr


def minmaxnormalize(mask_graph_x):
    if len(set(mask_graph_x['atom'].values())) == 1:
        return mask_graph_x
    
    data = np.array(list(mask_graph_x['atom'].values())).reshape(-1, 1)
    # data = (data/sum(data)).reshape(-1, )
    # mask_graph_x['atom'] = {k: v for k, v in enumerate(data)}
    normalized = StandardScaler().fit_transform(data)
    scaled = MinMaxScaler(feature_range=(0,1)).fit_transform(normalized).reshape(-1, )
    mask_graph_x['atom'] = {k: v for k, v in enumerate(scaled)}

    return mask_graph_x
    

def plot_attentions(args, sample_graph, sample_att):

    smiles = sample_graph.smiles
    mol = Chem.MolFromSmiles(smiles)

    if len(args.reduced) == 0:
        sample_att_g = sample_att[0]
        mask_graph_g = mask_graph(sample_att_g)
        
        fig = display_interpret_weight(mol, None, None, mask_graph_g, None, scale=True)

    else:
        reduced_graph, cliques, edges = getReducedGraph(args, args.reduced, smiles, normalize=False)
            
        sample_att_g, sample_att_r = sample_att
        if args.schema in ['A', 'AR', 'AR_0', 'AR_N']:
            mask_graph_g = mask_graph(sample_att_g)
        if args.schema in ['R', 'R_0', 'R_N', 'AR', 'AR_0', 'AR_N']:
            mask_graph_r = mask_reduced(sample_att_r)
            
        if args.schema in ['A', 'AR', 'AR_0', 'AR_N']:    
            fig = display_interpret_weight(mol, cliques, edges, mask_graph_g, mask_graph_r, scale=True)
        elif args.schema in ['R', 'R_0', 'R_N']:
            fig = display_interpret_weight(mol, cliques, edges, None, mask_graph_r, scale=True)

    return fig

def plot_attentions_groundtruth(args, sample_graph, sample_att):

    smiles = sample_graph.smiles
    mol = Chem.MolFromSmiles(smiles)

    sample_att_g = sample_att[0]
    mask_graph_g = mask_graph(sample_att_g)
    
    display_interpret_weight(mol, None, None, mask_graph_g, None, scale=True, color_map='Blues')



def plot_attentions_groundtruth_compare(mol, mask_graph_g, sample_graph, sample_att, auc):
    # formatting mol
    mol = mol_with_atom_index(mol)
    AllChem.Compute2DCoords(mol)
    
    fig, ax = plt.subplots(2, 1, figsize=(10,20))
    fig.suptitle('AttAUC: '+"{:.2f}".format(auc), fontsize=14, fontweight='bold')

    # Predicted
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Greens')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)

    inp_graph_predicted = display_interpret_graph(mol, mask_graph_g, scale=True, colors=plt_colors)
    
    # mol with sub
    ax1 = plt.subplot(2, 1, 1)
    ax1.grid(False)
    ax1.axis('off')
    ax1.imshow(inp_graph_predicted)

    # Ground Truth 
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Blues')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)

    sample_att_g = sample_att[0]
    mask_graph_g = mask_graph(sample_att_g)

    inp_graph_ground = display_interpret_graph(mol, mask_graph_g, scale=True, colors=plt_colors)
    
    # mol with sub
    ax2 = plt.subplot(2, 1, 2)
    ax2.grid(False)
    ax2.axis('off')
    ax2.imshow(inp_graph_ground)


def getReducedGraph(args, reduced, smiles, normalize):
    if 'atom' in reduced:
        reduced_graph = AtomGraph(smiles, normalize)
        # print(reduced_graph.node_features)
        cliques, edges = reduced_graph.cliques, reduced_graph.edges
    elif 'junctiontree' in reduced:
        reduced_graph = JunctionTreeGraph(smiles, normalize)
        # print(reduced_graph.node_features)
        cliques, edges = reduced_graph.cliques, reduced_graph.edges
    elif 'cluster' in reduced:
        reduced_graph = ClusterGraph(smiles, normalize)
        # print(reduced_graph.node_features)
        cliques, edges = reduced_graph.cliques, reduced_graph.edges
    elif 'functional' in reduced:
        reduced_graph = FunctionalGraph(smiles, normalize)
        # print(reduced_graph.node_features)
        cliques, edges = reduced_graph.cliques, reduced_graph.edges
    elif 'pharmacophore' in reduced:
        reduced_graph = PharmacophoreGraph(smiles, normalize)
        # print(reduced_graph.node_features)
        cliques, edges = reduced_graph.cliques, reduced_graph.edges
    elif 'substructure' in reduced:
        vocab_path = 'vocab/'+args.file+'_0.txt'
        tokenizer = Tokenizer(vocab_path)
        reduced_graph = SubstructureGraph(smiles, tokenizer, normalize)
        # print(reduced_graph.node_features)
        cliques, edges = reduced_graph.cliques, reduced_graph.edges
        
    return reduced_graph, cliques, edges

definedFunc = [fparams.GetFuncGroup(i).GetProp('_Name') for i in range(fparams.GetNumFuncGroups())]
temp = definedFunc[27]
definedFunc[27] = definedFunc[29]
definedFunc[29] = temp
definedFunc_interpret = definedFunc

definedFuncBond_interpret = []
for f in definedFuncBond:
    if f == 'Unknown':
        definedFuncBond_interpret.append('Unknown Bond')
    else:
        if f[1] == Chem.rdchem.BondType.SINGLE:
            bond = '-'
        elif f[1] == Chem.rdchem.BondType.DOUBLE:
            bond = '='
        elif f[1] == Chem.rdchem.BondType.TRIPLE:
            bond = '#'
        definedFuncBond_interpret.append(f[0]+bond+f[2])

definedRingOther = [r+'_atom' for r in definedRing]
definedRingSize = ['ring_'+r for r in ['3','4','5','6','7','8','9','>9']]
definedRing_interpret = definedRing+definedRingOther+definedRingSize

def getImportanceFeatures(reduced, f):
    f_list = [ff for ff in f]
    selected = list()
    features = list()
    if 'atom' in reduced:
        selected = list(range(0, 43+1))+[69, 78]
        features = definedAtom+['In Aromatic', 'In Ring']
    elif 'junctiontree' in reduced:
        selected = list(range(33, 82+1))
        features = ['Is Ring', 'Is Bond']+definedAtom+['Single', 'Double', 'Triple', 'Aromatic']
    elif 'cluster' in reduced:
        selected = list(range(33, 82+1))
        features = ['Is Ring', 'Is Bond']+definedAtom+['Single', 'Double', 'Triple', 'Aromatic']
    elif 'functional' in reduced:
        selected = list(range(0, 114+1))
        features = definedFunc_interpret+definedRing_interpret+definedFuncBond_interpret
    elif 'pharmacophore' in reduced:
        selected = list(range(0, 5+1))
        features = ['Is H-Donor',
                    'Is H-Acceptor',
                    'Is Positive',
                    'Is Negative',
                    'Is Hydrophobic',
                    'Is Aromatic']
    elif 'substructure' in reduced:
        selected = list(range(0, 1023+1))

    if 'substructure' not in reduced:
        f_list = [f_list[s] for s in selected]
        if reduced in ['junctiontree', 'cluster']:
            f_list = [features[i]+str(f) for i, f in enumerate(f_list) if float(f) != 0]
        else:
            f_list = [features[i] for i, f in enumerate(f_list) if float(f) != 0]

    if len(f_list) == 0:
        return '['+reduced[0][0]+'] '+str(tuple('-'))
    else:
        return '['+reduced[0][0]+'] '+str(tuple(f_list)) if len(f_list) > 1 else '['+reduced[0][0]+'] '+str(f_list[0])