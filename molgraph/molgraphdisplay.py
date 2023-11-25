######################
### Import Library ###
######################

# my library
from molgraph.molgraph import *
from molgraph.utilsonehot import *
from molgraph.utilsmol import *
from molgraph.utilsgraph import *
# standard
import os as os
import copy as copy
import random as random
# plotting
import matplotlib.pyplot as plt
from skimage.io import imread
from cairosvg import svg2png
# rdkit
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
rdDepictor.SetPreferCoordGen(True)

########################
### Display Function ###
########################

# display molecule graph
def display_molecule_graph(mol, mask_graph, scale=False, colors=None):
    
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
    dg.SetFontSize(6)
    
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    dg.DrawMolecule(mol, 
                    highlightAtoms=mask_graph_atom_key,
                    highlightAtomColors=mask_graph_color['atom'],
                    highlightBonds=mask_graph_bond_key,
                    highlightBondColors=mask_graph_color['bond'])
    
    dg.FinishDrawing()
    svg = dg.GetDrawingText()
    svg = svg.replace('svg:', '')
    svg = svg.replace("<rect style='opacity:1.0", "<rect style='opacity: 0")
    svg2png(bytestring=svg, write_to='tmp_graph.png', dpi=300, background_color='transparent')
    img = imread('tmp_graph.png')
    os.remove('tmp_graph.png')
    return img

# display molecule reduced graph
def display_molecule_reduced(mol, mask_graph, edges, mol_pos=None, scale=False, colors=None):
    
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
    
    plt.savefig('tmp_reduced.png', dpi=300, transparent=True)
    plt.close(fig)
    img = plt.imread('tmp_reduced.png')
    os.remove('tmp_reduced.png')
    return img
    

# display molecule as img of graph and reduced graph
def display_img(moleculegraph):
    mol = moleculegraph.mol
    cliques = moleculegraph.cliques
    edges = moleculegraph.edges

    # formatting mol
    mol = mol_with_atom_index(mol)
    AllChem.Compute2DCoords(mol)
    number_of_sub = len(cliques)
    
    # random colors
    colors = []
    for i in range(number_of_sub):
        r = random.randint(0,255)/255
        g = random.randint(0,255)/255
        b = random.randint(0,255)/255
        colors.append([r,g,b,1])
    
    # construct mask graph
    mask_graph = {}
    mask_graph['atom'] = {}
    mask_graph['bond'] = {}
    for i, c in enumerate(cliques):
        for a in c:
            # should have multiple
            # mask_graph['atom'].append({a: i})
            mask_graph['atom'][a] = i
            for j, b in enumerate(mol.GetBonds()):
                if all(bb in c for bb in [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]):
                    mask_graph['bond'][j] =  i
                    
    # construct mast subgraph
    mask_subgraph = {}
    mask_subgraph['atom'] = {}
    mask_subgraph['bond'] = {}
    for i, e in enumerate(cliques):
        mask_subgraph['atom'][i] = i
        mask_subgraph['atom'][i] = i
    
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

    img_graph = display_molecule_graph(mol, mask_graph, scale=False, colors=colors)
    img_reduced = display_molecule_reduced(mol, mask_subgraph, edges, mol_pos=mol_pos, scale=False, colors=colors)

    return img_graph, img_reduced


# display molecule alignment horizontal / vertical
def display_alignment(moleculegraph, alignment='horizontal'):
    img_graph, img_reduced = display_img(moleculegraph)

    # visualize
    if alignment == 'horizontal':
        fig, ax = plt.subplots(1, 2, figsize=(15,7))
        plt.clf()
    elif alignment == 'vertical':
        fig, ax = plt.subplots(2, 1, figsize=(7,15))
        plt.clf()
    
    # ax setting
    if alignment == 'horizontal':
        ax1 = plt.subplot(1, 2, 1)
    elif alignment == 'vertical':
        ax1 = plt.subplot(2, 1, 1)

    ax1.grid(False)
    ax1.axis('off')
    ax1.imshow(img_graph)

    # ax setting
    if alignment == 'horizontal':
        ax2 = plt.subplot(1, 2, 2)
    elif alignment == 'vertical':
        ax2 = plt.subplot(2, 1, 2)
    
    ax2.grid(False)
    ax2.axis('off')
    ax2.imshow(img_reduced)

    plt.tight_layout()
    fig.savefig('tmp_combine.png', dpi=300)
    plt.close(fig)
    img = plt.imread('tmp_combine.png')
    os.remove('tmp_combine.png')
    return img


# display molecule graph and reduced graph only one kind of graph
def display_one_graph(moleculegraph):
    fig, ax = plt.subplots(1, 1, figsize=(30,14))
    ax1 = plt.subplot(1, 1, 1)
    ax1.grid(False)
    ax1.axis('off')
    ax1.imshow(display_alignment(moleculegraph, alignment='horizontal'))


# display molecule all kinds of graphs
def display_all_graph(mol):
    fig, ax = plt.subplots(1, 5, figsize=(35,15))
    ax1 = plt.subplot(1, 5, 1)
    ax2 = plt.subplot(1, 5, 2)
    ax3 = plt.subplot(1, 5, 3)
    ax4 = plt.subplot(1, 5, 4)
    ax5 = plt.subplot(1, 5, 5)

    # atom-based original graph
    atom_graph = AtomGraph(mol)
    ax1.grid(False)
    ax1.axis('off')
    ax1.imshow(display_alignment(atom_graph, alignment='vertical'))

    # junctontree-based graph
    junctiontree_graph = JunctionTreeGraph(mol)
    ax2.grid(False)
    ax2.axis('off')
    ax2.imshow(display_alignment(junctiontree_graph, alignment='vertical'))

    # cluster-based graph
    cluster_graph = ClusterGraph(mol)
    ax3.grid(False)
    ax3.axis('off')
    ax3.imshow(display_alignment(cluster_graph, alignment='vertical'))

    # functional-based graph
    functional_graph = FunctionalGraph(mol)
    ax4.grid(False)
    ax4.axis('off')
    ax4.imshow(display_alignment(functional_graph, alignment='vertical'))

    # pharmacophore-based graph
    pharmacophore_graph = PharmacophoreGraph(mol)
    ax5.grid(False)
    ax5.axis('off')
    ax5.imshow(display_alignment(pharmacophore_graph, alignment='vertical'))

    plt.tight_layout()
    fig.savefig('tmp_all.png', dpi=300)
    os.remove('tmp_all.png')
