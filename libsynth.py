from collections import defaultdict
from itertools import chain, product
from IPython.display import SVG
import numpy as np
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools, rdDepictor, rdMolDescriptors, rdmolops
from rdkit.Chem.rdchem import EditableMol
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import DrawingOptions


# A bunch of helper functions

def perceive_rgroups(mol):
    """
    Read in a core molecule with R-groups marked with istope labels and move the R-group labels to atom maps.
    @param Chem.rdchem.Mol mol: input molecule
    @return: None
    """
    for atm in mol.GetAtoms():
        isotope = atm.GetIsotope()
        if isotope > 0:
            atm.SetAtomMapNum(isotope)
            atm.SetIsotope(0)
            
def perceive_list_atoms(mol):
    """
    Read in a core molecule with list atoms and mark the position with an atom map number.
    @param Chem.rdchem.Mol mol: input molecule
    @return: Dictionary of list atoms 
    """
    l_atoms = get_list_atoms(mol)
    l_atoms_new = {}
    if len(l_atoms) > 0:
        idxs = [*l_atoms]
        max_map_num = get_max_atom_map_num(mol) # highest R group atom map number
        for atm in mol.GetAtoms():
            if atm.GetIdx() in idxs:
                max_map_num = max_map_num + 1
                atm.SetAtomMapNum(max_map_num) # set atom map number for list atom here
                l_atoms_new[max_map_num] = l_atoms[atm.GetIdx()]
                
    for key, values in l_atoms_new.items(): # set atom map numbers for individual list atoms
        for value in values:
            value.SetAtomMapNum(key)
            
    return l_atoms_new
                
def get_max_atom_map_num(mol):
    """
    Read in a core molecule with R groups and returns the max atom map number.
    @param Chem.rdchem.Mol mol: input molecule
    @return: Maximum atom map number.
    """
    max_map_num = 0
    for atm in mol.GetAtoms():
        map_num = atm.GetAtomMapNum()
        if map_num > max_map_num:
            max_map_num = map_num
    return max_map_num
            
def get_attachpoint_atom(mol):
    """
    Read in a reagent molecule and return the index of the attach point atom.
    @param Chem.rdchem.Mol mol: input molecule
    @return: Index of attach point atom.
    """
    attach_atom = -1
    for a in mol.GetAtoms():
        if a.HasProp('molAttchpt'):
            attach_atom = a.GetIdx()
    return attach_atom

def peel_reagent(mol, rgp_idx):
    """
    Read in a reagent molecule and prepare it for attachment with core molecule by peeling terminal atoms.
    @param Chem.rdchem.Mol mol: input molecule
    @param int rgp_idx: the R groups isotope number (1, 2, 3, etc.) 
    @return: Reagent molecule prepared for attachment with the core.
    """
    name = mol.GetProp('Name') # reagent file name without file extension
    atomsToRemove = []
    attch_idx = get_attachpoint_atom(mol)
    ri = mol.GetRingInfo() # to check ring membership - if atom in ring, do not remove atom
    if attch_idx >= 0: # make sure that the reagent has attach point, else throw error
        for a in mol.GetAtoms():
            if a.GetAtomMapNum() >0: # for atoms with a mapping
                if a.GetIdx() == attch_idx: # if the atom is attach point
                    a.SetAtomMapNum(rgp_idx)
                else: # other atoms to be removed
                    a.SetAtomMapNum(0)
                    rings_involved = ri.NumAtomRings(a.GetIdx())
                    if rings_involved == 0:
                    	atomsToRemove.append(a.GetIdx())
                
    else:
        print(f'No attachment point found for reagent \'{name}\'') # throw error - reagent does not have attachment point
        return None

    
    em = Chem.EditableMol(mol)
    
    if atomsToRemove:
        atomsToRemove.sort(reverse=True)
        for atom in atomsToRemove:
            em.RemoveAtom(atom)
    else:
        print('Cannot find terminal atoms to remove.')
    
    peeled_mol = em.GetMol()
    try:
    	final_mol = Chem.RemoveHs(peeled_mol)
    	if final_mol:
    		Chem.SanitizeMol(final_mol)
    		final_mol.SetProp('RGP_ID', rgp_idx) # setting the R-group isotope num as a property for later use
    		return final_mol
    	else:
    		return None
    except Exception as e:
    	print(f'Error reading reagent \'{name}\': {e}')
    	return None

    
def getMolsFromDir(path):
    """
    Read in path to a directory and reads all MOL files.
    @param str path: input directory
    @return: List of molecules.
    """
    mols = []
    for root, directories, files in os.walk(path, topdown=False):
        for fname in files:
            if fname.endswith('.mol'):
            	mol = Chem.MolFromMolFile(os.path.join(root, fname))
            	mol.SetProp('Name', fname.split('.')[0])
            	mols.append(mol)
    return mols

def get_matching_atoms(mol, reagent):
    """
    Reads in query and a reagent molecule and returns a list of atom indices from the query molecule that
    match the reagent. If multiple matches with overlapping atoms are possible, a unique list of atom indices
    is returned. If no macthes were found, an empty list is returned.
    @param Chem.rdchem.Mol mol: query molecule
    @param Chem.rdchem.Mol reagent: reagent
    @return: Set of matches each with a list of atom indices.
    """
    
    matches = mol.GetSubstructMatches(reagent)
    unique_match = []
    for m in list(matches):
        for n in m:
            unique_match.append(n)
    
    return unique_match

def get_matching_atoms_first(mol, reagent):
    """
    Reads in query and a reagent molecule and returns a list of atom indices from the query molecule that
    match the reagent. If multiple matches with overlapping atoms are possible, a unique list of atom indices
    is returned. If no macthes were found, an empty list is returned.
    @param Chem.rdchem.Mol mol: query molecule
    @param Chem.rdchem.Mol reagent: reagent
    @return: List of atom indices.
    """
    
    match = mol.GetSubstructMatch(reagent)
    
    return list(match)

def get_list_atoms(mol):
    """
    Reads in a molecule and returns a dictionary with atom indices as keys and corresponding list atoms 
    (if any) as values.
    @param Chem.rdchem.Mol mol: input molecule
    @return: list atoms stored in a dictionary and indexed by atom indices of the parent molecule
    """
    atom_map = {}
    for a in mol.GetAtoms():
        if a.HasQuery():
            if a.GetIsotope() == 0: # this is a not yet a list of atoms
                sm = a.GetSmarts()
                sm = sm[1:-1] # stripping the square backets
                sm_atms = sm.split(",")
                if len(sm_atms) > 1: # now, this is a list of atoms
                    list_atms = []
                    for atm in sm_atms:
                        list_atm = Chem.AtomFromSmarts('[' + atm + ']')
                        #list_atm.SetIsotope(a.GetIdx()) # retaining the atom index via Isotope as setIdx() cannot be used for QueryAtom
                        list_atms.append(list_atm)
                    if list_atms:
                        atom_map[a.GetIdx()] = list_atms
    return atom_map

def get_expected_size(attachments):
    """
    Reads in a dictionary of attachments and provides the estimated size of the enumerated virtual library.
    @param dict attachments: A dictionary of reagents (attachment) where key are R group atom isotope numbers and values
    are the list of reagents for each isotope number.
    @return: Expected size of the virtual library.
    """
    size = 0
    for key, val in attachments.items():
        if size == 0:
            size = len(val)
        else:
            size *= len(val)
    return size

FUNCS = {name: func for name, func in Descriptors.descList} # borrowed from https://github.com/bp-kelley/descriptastorus

desc_names = ['MolWt', 'NumHDonors', 'NumHAcceptors', 'MolLogP', 'TPSA', 'MolMR', 'BertzCT']

def apply_func(name, mol): # borrowed from https://github.com/PatWalters/useful_rdkit_utils
    """Apply an RDKit descriptor calculation to a moleucle
    :param name: descriptor name
    :param mol: RDKit molecule
    :return:
    """
    try:
        return FUNCS[name](mol)
    except:
        return None

def calc_desc(mol): # borrowed from https://github.com/PatWalters/useful_rdkit_utils
        """Calculate descriptors for an RDKit molecule
        :param mol: RDKit molecule
        :return: a numpy array with descriptors
        """
        res = [apply_func(name, mol) for name in desc_names]
        return np.array(res, dtype=float)


# R-group attachment - logic 1 (core molecule and the reagents both have R-atoms)

def attach_rgroup_two_sided(target, chain):
    """
    Reads in core and reagent molecules with an R group atom each and the isotope labels moved to the atom maps.
    In this case, an attachment is made only if the atom map numbers from the core and reagent match.
    @param Chem.rdchem.Mol: input molecule
    @return: Core molecule with reagent attached.
    """
    # combine mols
    newmol = Chem.RWMol(rdmolops.CombineMols(target, chain))
    atoms = newmol.GetAtoms()
    mapper = defaultdict(list)
    for idx, atm in enumerate(atoms):
        atom_map_num = atm.GetAtomMapNum()
        if atom_map_num > 0:
            mapper[atom_map_num].append(idx)
            
    for idx, a_list in mapper.items():
        if len(a_list) == 2:
            atm1, atm2 = a_list
            rm_atoms = [newmol.GetAtomWithIdx(atm1),newmol.GetAtomWithIdx(atm2)]
            nbr1 = [x.GetOtherAtom(newmol.GetAtomWithIdx(atm1)) for x in newmol.GetAtomWithIdx(atm1).GetBonds()][0]
            nbr1.SetAtomMapNum(idx)
            nbr2 = [x.GetOtherAtom(newmol.GetAtomWithIdx(atm2)) for x in newmol.GetAtomWithIdx(atm2).GetBonds()][0]
            nbr2.SetAtomMapNum(idx)
    newmol.AddBond(nbr1.GetIdx(), nbr2.GetIdx(), order=Chem.rdchem.BondType.SINGLE)
    nbr1.SetAtomMapNum(0)
    nbr2.SetAtomMapNum(0)
    newmol.RemoveAtom(rm_atoms[0].GetIdx())
    newmol.RemoveAtom(rm_atoms[1].GetIdx())
    
    final_mol = newmol.GetMol()
    final_mol = Chem.RemoveHs(final_mol)
    Chem.SanitizeMol(final_mol)
    AllChem.Compute2DCoords(final_mol)

    return final_mol

# R-group attachment - logic 2 (core molecule has R-atoms but the reagents only have sticky attachment points)

def attach_rgroup_one_sided(target, chain):
    """
    Reads in core and reagent molecules where core has an R group atom but reagent only has the atom map set.
    In this case, an attachment is made only if the atom map numbers from the core and reagent match.
    @param Chem.rdchem.Mol: input molecule
    @return: Core molecule with reagent attached.
    """
    # combine mols
    rgp_id = chain.GetProp('RGP_ID')
    rgp_smiles_prop = 'RGP_ID_' + rgp_id
    newmol = Chem.RWMol(rdmolops.CombineMols(target, chain))
    atoms = newmol.GetAtoms()
    mapper = defaultdict(list)
    
    for idx, atm in enumerate(atoms):
        atom_map_num = atm.GetAtomMapNum()
        if atom_map_num > 0:
            mapper[atom_map_num].append(idx)
            
    for idx, a_list in mapper.items():
        if len(a_list) == 2:
            atm1, atm2 = a_list
            rm_atoms = [newmol.GetAtomWithIdx(atm1)]
            nbr1 = [x.GetOtherAtom(newmol.GetAtomWithIdx(atm1)) for x in newmol.GetAtomWithIdx(atm1).GetBonds()][0]
            nbr1.SetAtomMapNum(idx)
            nbr2 = newmol.GetAtomWithIdx(atm2)   
    newmol.AddBond(nbr1.GetIdx(), nbr2.GetIdx(), order=Chem.rdchem.BondType.SINGLE)
    nbr1.SetAtomMapNum(0)
    nbr2.SetAtomMapNum(0)
    newmol.RemoveAtom(rm_atoms[0].GetIdx())

    final_mol = newmol.GetMol()
    final_mol = Chem.RemoveHs(final_mol)
    final_mol.SetProp(rgp_smiles_prop, Chem.MolToSmiles(chain))
    Chem.SanitizeMol(final_mol)
    AllChem.Compute2DCoords(final_mol)

    return final_mol

# List atoms attachment logic (the list atom's neighbor atom from the core and the individual list atoms have complementary atom mappings)

def attach_latom_one_sided(target, chain): # a variant of above method where 
    """
    Reads in core and list atom molecules where both have the atom map set. In this case, an attachment is 
    made only if the atom map numbers from the core and list atom match.
    @param Chem.rdchem.Mol: input molecule
    @return: Core molecule with list atom attached.
    """
    # combine mols
    newmol = Chem.RWMol(rdmolops.CombineMols(target, chain))
    atoms = newmol.GetAtoms()
    mapper = defaultdict(list)
    
    for idx, atm in enumerate(atoms):
        atom_map_num = atm.GetAtomMapNum()
        if atom_map_num > 0:
            mapper[atom_map_num].append(idx)
            
    for idx, a_list in mapper.items():
        if len(a_list) == 2:
            atm1, atm2 = a_list
            atom1 = newmol.GetAtomWithIdx(atm1)
            atom2 = newmol.GetAtomWithIdx(atm2)
            newmol.AddBond(atom1.GetIdx(), atom2.GetIdx(), order=Chem.rdchem.BondType.SINGLE)
            atom1.SetAtomMapNum(0)
            atom2.SetAtomMapNum(0)

    final_mol = newmol.GetMol()
    final_mol = Chem.RemoveHs(final_mol)
    Chem.SanitizeMol(final_mol)
    AllChem.Compute2DCoords(final_mol)

    return final_mol

def attach_list_atom(target, atom):
    
    # 1. transfer the atom map on the list atom to the neighbor atom on core mol
    # 2. then remove the list atom from the core mol
    # 3. make bond with the atom that is supplied as argument, but first transform QueryAtom into Mol
    
    map_num = atom.GetAtomMapNum()
    for atm in target.GetAtoms():
        if atm.GetAtomMapNum() == map_num:
            nbr_atom = [x.GetOtherAtom(atm) for x in atm.GetBonds()][0]
            nbr_atom.SetAtomMapNum(map_num)
            # prepare the core mol and list atom mol
            atm.SetAtomicNum(0) #
            target = Chem.DeleteSubstructs(target, Chem.MolFromSmarts('[#0]')) # core prepared
            list_atom_mol = Chem.MolFromSmarts(atom.GetSmarts()) # list atom mol prepared
            final_mol = attach_latom_one_sided(target, list_atom_mol)

            return final_mol

# enumeration

def enumerate_virtual_library(core: Chem.rdchem.Mol, attachments: dict):
    enum_mols = []
    attachments_sorted = sorted(attachments)
    combs = product(*(attachments[x] for x in attachments_sorted)) # all possible combinations of attachments
    for i in combs:
        mol = Chem.Mol(core) # each time, use a new copy of the core mol
        for j, attachment in enumerate(i):
            if type(attachment) is Chem.rdchem.Mol: # r group attachment happens here
                mol = attach_rgroup_one_sided(mol, attachment)
            elif type(attachment) is Chem.rdchem.QueryAtom: # list atom attachment happens here
                mol = attach_list_atom(mol, attachment)
        matches = []
        for j, reagent in enumerate(i):
            if type(reagent) is Chem.rdchem.Mol:
                match = get_matching_atoms(mol,reagent)
                if match:
                    matches.extend(match)
        matches_unique = list(set(matches))
        if matches_unique:
            matches_str = ",".join([str(int) for int in matches_unique])
            mol.SetProp("RAtomDict", matches_str)
        else:
            print('No Matches')
            mol.SetProp("RAtomDict", 'NA')
        enum_mols.append(mol)
    return enum_mols
