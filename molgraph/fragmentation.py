# SAPpy
from util.sarpy.SARpy import *
from util.sarpy.SARpytools import *
# rdkit
from rdkit import Chem
from rdkit.Chem import Recap
from rdkit.Chem import BRICS

# Functions
# def recap_frag_smiles_leaf(mols):
#     all_leaves=set()
#     for mol in mols:
#         tree = Recap.RecapDecompose(mol)
#         leaves = tree.GetLeaves().keys()
#         all_leaves.update(leaves)
#     return all_leaves

def recap_frag_smiles_children(mols, limit=[3,25]):
    all_children=set()
    for mol in mols:
        Chem.RemoveStereochemistry(mol) 
        hierarch = Recap.RecapDecompose(mol)
        children = hierarch.GetAllChildren().keys()
        # children = hierarch.GetLeaves().keys()

        for c in children:
            c_smiles = c
            children_mol = Chem.MolFromSmarts(c_smiles)

            # limit number of atom
            any_atom = 0
            for a in children_mol.GetAtoms():
                if a.GetSymbol() != '*':
                    any_atom += 1
            if any_atom < limit[0] or any_atom > limit[1]:
                continue
            
            # display(mol)
            # print('BEFORE:')
            # print(c_smiles)
            # display(children_mol)

            # assign UNSPECIFIED bond to * atom
            for b in children_mol.GetBonds():
                if b.GetBeginAtom().GetSymbol() == '*':
                    b.SetBondType(Chem.rdchem.BondType.UNSPECIFIED)
                if b.GetEndAtom().GetSymbol() == '*':
                    b.SetBondType(Chem.rdchem.BondType.UNSPECIFIED)

            c_smiles = Chem.MolToSmiles(children_mol)
            children_mol = Chem.MolFromSmarts(c_smiles)
            # Chem.Kekulize(children_mol, clearAromaticFlags=True)
            
            # print('AFTER:')
            # print(c_smiles)
            # display(children_mol)

            # if len(mol.GetSubstructMatch(children_mol))==0:
            #     print(Chem.MolToSmiles(mol), c_smiles)

            all_children.update([c_smiles])

    return all_children

def brics_frag_smiles(mols, limit=[3,25]):
    all_frags = set()
    for mol in mols:
        Chem.RemoveStereochemistry(mol) 
        frags = BRICS.BRICSDecompose(mol,keepNonLeafNodes=True)

        for f in frags:
            f_smarts = f
            frag_mol = Chem.MolFromSmarts(f_smarts)

            # limit number of atom
            any_atom = 0
            for a in frag_mol.GetAtoms():
                if a.GetSymbol() != '*':
                    any_atom += 1
            if any_atom < limit[0] or any_atom > limit[1]:
                continue

            # display(mol)
            # print('BEFORE:')
            # print(f_smarts)
            # display(frag_mol)

            for a in frag_mol.GetAtoms():
                if a.GetSymbol() == '*':
                    a.SetIsotope(0)
            for b in frag_mol.GetBonds():
                if b.GetBeginAtom().GetSymbol() == '*':
                    b.SetBondType(Chem.rdchem.BondType.UNSPECIFIED)
                if b.GetEndAtom().GetSymbol() == '*':
                    b.SetBondType(Chem.rdchem.BondType.UNSPECIFIED)

            f_smiles = Chem.MolToSmiles(frag_mol)
            frag_mol = Chem.MolFromSmarts(f_smiles)

            # print('AFTER:')
            # print(f_smiles)
            # display(frag_mol)

            # if len(mol.GetSubstructMatch(frag_mol))==0:
            #     print(Chem.MolToSmiles(mol), f_smiles)

            all_frags.update([f_smiles])

    return all_frags

def grinder_frag_smiles(mols, limit=[3,25]):
    all_frags = set()

    grinder = Grinder(limit[0], limit[1])
    for mol in mols:
        Chem.RemoveStereochemistry(mol) 
        smiles = Chem.MolToSmiles(mol)
        pb_mol = pb.readstring("smi",smiles)
        pb_struct = Structure(pb_mol)
        fragments = collectSubs([pb_struct], grinder)

        for f in range(len(fragments)):
            f_smarts = fragments[f].smiles
            frag_mol = Chem.MolFromSmarts(f_smarts)

            # assign UNSPECIFIED bond to * atom
            for b in frag_mol.GetBonds():
                if b.GetBeginAtom().GetSymbol() == '*':
                    b.SetBondType(Chem.rdchem.BondType.UNSPECIFIED)
                if b.GetEndAtom().GetSymbol() == '*':
                    b.SetBondType(Chem.rdchem.BondType.UNSPECIFIED)

            # display(mol)
            # print('BEFORE:')
            # print(f_smarts)
            # display(frag_mol)

            f_smiles = Chem.MolToSmiles(frag_mol)
            frag_mol = Chem.MolFromSmarts(f_smiles)

            # print('AFTER:')
            # print(f_smiles)
            # display(frag_mol)

            # if len(mol.GetSubstructMatch(frag_mol))==0:
            #     print(Chem.MolToSmiles(mol), f_smiles)

            all_frags.update([f_smiles])

    return all_frags