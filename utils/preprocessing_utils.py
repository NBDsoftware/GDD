import os
import subprocess
import time
import prody
import warnings
import torch
import errno
import math
import csv
import logging
import copy
import json
import ipdb
import shutil

import Bio
import numpy as np
import networkx as nx

from datasets.process_mols import read_molecule
from utils.featurization import featurize_mol
from utils.torsion import (
    AngleCalcMethod,
    TorsionCalculator,
)
from utils.guidance import get_rot_state
from utils.metrics import get_rot_angles

from numbers import Number
from scipy import spatial
from urllib.request import urlopen
from torch_geometric.utils import to_networkx
from networkx.algorithms import isomorphism

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.rdchem import AtomValenceException, Mol
from rdkit.Chem.MolStandardize import rdMolStandardize

from Bio.PDB import Select, PDBParser, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from utils.exceptions import *

from dataclasses import dataclass

###########################################
###### HELPER FUNCTIONS AND CLASSES #######
###########################################


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class ResidueSelect(Select):
    def __init__(self, model, chain, residue):
        self.model = model
        self.chain = chain
        self.residue = residue

    def accept_model(self, model):
        return model.id == self.model.id

    def accept_chain(self, chain):
        return chain.id == self.chain.id

    def accept_residue(self, residue):
        """Recognition of heteroatoms - Remove water molecules"""
        return residue == self.residue and is_het(residue)


class NonHetSelect(Select):
    def __init__(self, model):
        self.model = model

    def accept_model(self, model):
        return model.id == self.model.id

    def accept_residue(self, residue):
        return not is_het(residue)


class NoHetsOrWaterSelect(Select):
    def __init__(self, model, chain_id=None):
        self.model = model
        self.chain_id = chain_id

    def accept_model(self, model):
        return model == self.model
        
    def accept_chain(self, chain):
        if self.chain_id is None:
            return True
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return not is_het(residue) and residue.get_resname() != "HOH"


def is_het(residue):
    res = residue.id[0]
    return res != " " and res != "W"


def write_to_json(
    filename, 
    data, 
    method=None, 
    reorder=None,
    replace_existing=True,
):
    # Load existing data or initialize an empty structure
    if method is not None and reorder is not None:
        if os.path.exists(filename) and not replace_existing:
            with open(filename, "r") as file:
                json_data = json.load(file)
        else:
            json_data = {"true": {}, "false": {}}

        reorder_key = str(reorder).lower()
        json_data[reorder_key][method] = data

    # Write the updated data to the file
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def residue_molecular_weight(residue: Bio.PDB.Residue.Residue) -> float:
    """
    Calculate molecular weight of a PDB residue using RDKit.
    """
    pt = Chem.GetPeriodicTable()
    weight = 0
    for atom in residue.get_atoms():
        atomic_symbol = atom.element.title()  # covert to title case
        weight += pt.GetAtomicWeight(atomic_symbol)

    return weight


# def calculate_residue_mass(residue):
#     """Calculate the molecular weight of a Bio.PDB.Residue.Residue object."""
#     mass = 0.0
#     for atom in residue:
#         atom_name = atom.element  # Get the element name of the atom
#         mass += IUPACData.atomic_weights[atom_name]  # Add the atomic mass
#     return mass


def TD_get_transformation_mask(pyg_data):
    G = to_networkx(pyg_data, to_undirected=False)
    to_rotate = []
    edges = pyg_data.edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i + 1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate


def assign_bond_orders_from_template(ligand, ground_ligand):

    try:
        ground_ligand = AllChem.AssignBondOrdersFromTemplate(ligand, ground_ligand)

    except ValueError:
        # Some hydrogens don't get properly removed. Force removal of all hydrogens.

        removeHs_params = rdmolops.RemoveHsParameters()
        removeHs_params.removeDefiningBondStereo = True

        try:
            ligand = Chem.RemoveHs(ligand, removeHs_params)
            ground_ligand = Chem.RemoveHs(ground_ligand, removeHs_params)

        except AtomValenceException:
            # some ligands die after enforcing this Hs removal
            raise InvalidLigandAfterHsRemoval

        removeHs_params.removeDefiningBondStereo = False  # reset

        # match again
        try:
            ground_ligand = AllChem.AssignBondOrdersFromTemplate(ligand, ground_ligand)

        except ValueError:
            try:
                
                diff = ligand.GetNumAtoms() - ground_ligand.GetNumAtoms()
                
                template_smiles = Chem.MolToSmiles(ground_ligand)
                template = Chem.MolFromSmiles(template_smiles)

                # canonicalize to get the same atom order
                canon_smiles_ligand = Chem.MolToSmiles(ligand, canonical=True)
                canon_smiles_template = Chem.MolToSmiles(template, canonical=True)
                ligand = Chem.MolFromSmiles(canon_smiles_ligand)
                template = Chem.MolFromSmiles(canon_smiles_template)

                equal = False
                while not equal:
                    for idx, (ligand_atom, template_atom) in enumerate(
                        zip(ligand.GetAtoms(), template.GetAtoms())
                    ):
                        if ligand_atom.GetSymbol() != template_atom.GetSymbol():
                            ligand.GetAtomWithIdx(idx).SetAtomicNum(1)
                            break
                    equal = True

               
                sm = Chem.MolToSmiles(ligand)
                ligand = Chem.MolFromSmiles(sm)

                ground_ligand = AllChem.AssignBondOrdersFromTemplate(
                    ligand, ground_ligand
                )
            except:
                print(f"Mismatch: {diff}")
                raise NumberOfAtomMismatchError
    except:
        print(f"Who knows what failed")
        raise UnkownError

    return ligand, ground_ligand


def load_smiles_and_pdb(ligand_smiles, ligand_path) -> tuple:
    """Loads smiles and pdb/sdf file into molecules"""
    print(ligand_path)
    ligand = Chem.rdmolfiles.MolFromSmiles(ligand_smiles)
    try:
        ligand = Chem.RemoveAllHs(ligand)
    except Exception as e:
        print(e)
        ligand = Chem.rdmolfiles.MolFromSmiles(ligand_smiles, sanitize=False)
        ligand = Chem.RemoveAllHs(ligand, sanitize=False)

    if ligand_path.endswith(".pdb"):
        ground_ligand = Chem.rdmolfiles.MolFromPDBFile(ligand_path)
        try:
            ground_ligand = Chem.RemoveAllHs(ground_ligand)
        except Exception as e:
            print(e)
            ground_ligand = Chem.rdmolfiles.MolFromPDBFile(ligand_path, sanitize=False)
            ground_ligand = Chem.RemoveAllHs(ground_ligand, sanitize=False)

    elif ligand_path.endswith(".sdf"):
        mol2_fileName = ligand_path.replace("sdf", "mol2")
        removeHs_params = rdmolops.RemoveHsParameters()
        removeHs_params.removeDefiningBondStereo = True

        try:
            ground_ligand = Chem.MolFromMolFile(
                ligand_path, sanitize=False, removeHs=True
            )
        except OSError:
            raise OSError

        problem = False
        try:
            Chem.SanitizeMol(ground_ligand)
            ground_ligand = Chem.RemoveHs(ground_ligand, removeHs_params)
        except Exception as e:
            problem = True
        if problem:
            ground_ligand = Chem.MolFromMol2File(
                mol2_fileName, sanitize=False, removeHs=True
            )
            try:
                Chem.SanitizeMol(ground_ligand)
                ground_ligand = Chem.RemoveHs(ground_ligand, removeHs_params)
                problem = False
            except Exception as e:
                raise LigandFileLoadingError

    return ligand, ground_ligand


def reorder_mol(mol: Mol) -> Mol:
    """Reorders a rdkit molecule by converting to smiles and back to mol.

    If mol has a conformer, the conformer positions are updated as well."""

    mol_ = copy.deepcopy(mol)
    reordered_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    if len(mol.GetConformers()) >= 1:
        mapping = reordered_mol.GetSubstructMatch(mol_)
        reordered_to_original = {v: k for k, v in enumerate(mapping)}
        original_to_reordered = {v: k for k, v in reordered_to_original.items()}

        conf = mol_.GetConformer()  # old conformer
        new_conf = Chem.Conformer(reordered_mol.GetNumAtoms())
        for idx_orig in range(reordered_mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(idx_orig)
            idx_reord = original_to_reordered[idx_orig]
            new_conf.SetAtomPosition(idx_reord, pos)

        reordered_mol.AddConformer(new_conf)

    return reordered_mol


#####################################
###### PREPROCESSING FUNCTIONS ######
#####################################


def get_ligand_center(sdf_path: str) -> np.ndarray:
    """Get the geometric center of the ligand from SDF file"""
    mol = Chem.SDMolSupplier(sdf_path)[0]
    if mol is None:
        raise ValueError(f"Could not read molecule from {sdf_path}")
    
    # Get 3D coordinates - mol.GetConformer().GetPositions() returns numpy array
    coords = mol.GetConformer().GetPositions()
    return np.mean(coords, axis=0)

def get_chain_center(chain) -> np.ndarray:
    """Calculate geometric center of a chain"""
    coords = []
    for residue in chain:
        for atom in residue:
            coords.append(atom.get_coord())
    return np.mean(coords, axis=0)

def find_closest_chain(pdb_structure, ligand_center: np.ndarray):
    """Find the chain closest to the ligand center"""
    min_dist = float('inf')
    closest_chain = None
    
    for model in pdb_structure:
        for chain in model:
            chain_center = get_chain_center(chain)
            dist = np.linalg.norm(chain_center - ligand_center)
            if dist < min_dist:
                min_dist = dist
                closest_chain = chain
    
    return closest_chain, min_dist

def separate_pdb_ligands(
    pdb_file: str, 
    results_path: str, 
    desired_chain: str = None,
    ligand_file: str = None, 
    complex_name: str = None,
) -> None:
    """
    Given a pdb file, removes headers and heteroatoms and saves the protein and ligands
    independetly in a given directory.

    Keeps only the ligands with mw > 150
    """
    try:
        protein_name, ligand_name = complex_name.split('_')
    except ValueError:
        protein_name = complex_name
        ligand_name = complex_name + '_ligand'
        
    pdb = PDBParser().get_structure(protein_name, pdb_file)
    io = PDBIO()
    io.set_structure(pdb)
    
    # If ligand_file is provided, just save the protein w/0 hetatms
    if ligand_file:     
        complex_path = os.path.join(results_path, complex_name)
        os.makedirs(complex_path, exist_ok=True)
        
        # Save protein structure without heteroatoms
        protein_save_path = os.path.join(complex_path, f'{protein_name}_protein.pdb')
        io.save(protein_save_path, NoHetsOrWaterSelect(pdb[0]))  # assuming first model
        
        return
    
    for model in pdb:
        for chain in model:
            if desired_chain is not None and chain.id != desired_chain:
                continue
            
            for residue in chain:
                if not is_het(residue):
                    continue
            
                mw = residue_molecular_weight(residue)
                if mw <= 150:
                    continue

                res_name = residue.get_resname()
                # create a new folder for the protein-ligand complex
                complex_path = os.path.join(
                    results_path, complex_name, protein_name + "_" + res_name
                )
                os.makedirs(complex_path, exist_ok=True)
                
                # save ligand
                io.save(
                    os.path.join(complex_path, res_name + ".pdb"),
                    ResidueSelect(model, chain, residue),
                )
                # save protein
                io.save(
                    os.path.join(complex_path, f"{protein_name}_protein.pdb"),
                    NoHetsOrWaterSelect(model),
                )
                


def process_obabel(
    protein_path: str,
    output_path: str = None,
    remove: bool = False,
    print_return_code: bool = False,
    time_process: bool = False,
) -> None:
    """Processes a protein with openbabel

    Parameters
    ----------
    protein_path : str
        path to the pdb file of the protein
    output_path : str, optional
        name for the processed file, by default None.
        if not provided or None, the name will be the original protein with _obabel appended
    remove : bool, optional
        control whether to remove the original protein, by default True
    print_return_code : bool, optional
        control whether to print the return code of running obabel, by default False
    time_process : bool, optional
        control whether to print the time it took to do the processing, by default False
    """

    start_time = time.time()
    if output_path == None:
        basedir, pdb_file = os.path.split(protein_path)
        pdb_id, _ = os.path.splitext(pdb_file)
        output_path = os.path.join(basedir, pdb_id + "_obabel.pdb")

    return_code = subprocess.run(f"obabel {protein_path} -O{output_path}", shell=True)

    if print_return_code:
        print(return_code)

    if time_process:
        print("--- %s seconds ---" % (time.time() - start_time))

    if remove:
        subprocess.run(
            f"rm {protein_path} && mv {output_path} {protein_path}", shell=True
        )


def process_reduce():
    """processes a protein with reduce"""

    raise NotImplementedError


def select_protein_chains(
    protein_name: str,
    protein_path: str, 
    ligand_path: str, 
    cutoff: int = 10
) -> str:
    """Given a protein pdb and a ligand, it keeps the chains that are withing
    the given cutoff (in amstrongs)"""

    basedir, _ = os.path.split(protein_path)
    # protein_name = protein_file[:6]
    io = PDBIO()
    biopython_parser = PDBParser()

    lig = read_molecule(ligand_path, sanitize=True, remove_hs=False)
    if lig == None:
        print(f"ligand was none for {protein_path}")
        return None

    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure("random_id", protein_path)
        rec = structure[0]

    min_distances = []
    coords = []
    valid_chain_ids = []
    lengths = []

    
    for i, chain in enumerate(rec):
        chain_coords = []
        chain_is_water = False
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                chain_is_water = True
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))
            if (
                c_alpha != None and n != None and c != None
            ):  ## only append residue if it is an amino acid and not some weird molecule that is part of the complex
                chain_coords.append(np.array(residue_coords))
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)
        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0)
            distances = spatial.distance.cdist(lig_coords, all_chain_coords)
            min_distance = distances.min()
        else:
            min_distance = np.inf

        if chain_is_water:
            min_distances.append(np.inf)
        else:
            min_distances.append(min_distance)
        lengths.append(count)
        coords.append(chain_coords)
        if min_distance < cutoff and not chain_is_water:
            valid_chain_ids.append(chain.get_id())
    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0:
        valid_chain_ids.append(np.argmin(min_distances))

    valid_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())

    prot = prody.parsePDB(protein_path)
    sel = prot.select(" or ".join(map(lambda c: f"chain {c}", valid_chain_ids)))
    processed_protein_path = os.path.join(
        basedir, f"{protein_name}_protein_processed.pdb"
    )
    try:
        prody.writePDB(processed_protein_path, sel)
        return processed_protein_path
    except:
        raise NoChainsWithinCutoff


def get_ground_pos_func(
    protein_name: str,
    ligand_name: str,
    ligand_smiles: str,
    ligand_file: str,
    results_dir: str = None,
    draw: bool = True,
    images_path: str = None,
    print_info: bool = False,
    create_files: bool = False,
    R: Number = 7,
    tr_gamma: float = 0.2,
    theta_margin: float = 0.15,
    phi_margin: float = 0.15,
    rot_gamma: float = 0.2,
    tor_margin: float = 0.15,
    tor_gamma: float = 0.2,
    reordering: bool = True,
    angle_calc_method: AngleCalcMethod = AngleCalcMethod.TOR_CALC_2,
) -> None:

    basedir, _ = os.path.split(ligand_file)  
    complex = os.path.split(basedir)[1]
    print("Processing complex ", complex)

    ### LOAD BOTH LIGANDS ###

    try:
        ligand, ground_ligand = load_smiles_and_pdb(ligand_smiles, ligand_file)
    except LigandFileLoadingError as e:
        logging.error(
            f"LigandFileLoadingError: Error loading the sdf/mol2 file for the ligand {complex}"
        )
        return
    except OSError as e:
        logging.error(f"OSError, bad ligand for {complex}")
        return

    if reordering:
        ligand, ground_ligand = reorder_mol(ligand), reorder_mol(ground_ligand)

    try:
        ligand, ground_ligand = assign_bond_orders_from_template(ligand, ground_ligand)
    except NumberOfAtomMismatchError as e:
        logging.error(
            f"NumberOfAtomsMismatchError: Ligand and ground ligand don't have the same number of atoms for complex {complex}"
        )
        return
    except InvalidLigandAfterHsRemoval as e:
        logging.error(
            f"InvalidLigandAfterHsRemoval: While attempting to remove Hs to fix a mismatch of number of atoms, ligand got messed up and rdkit can't load it for complex {complex}"
        )
        return
    except UnkownError as e:
        logging.error(f"UnkownError: {complex}")
        return

    ### FEATURIZE MOLS ###

    try:
        ground_data = featurize_mol(ground_ligand)
    except:
        logging.error(f"Featurize mol failed for ground ligand")
        return

    ground_data.mol = ground_ligand
    ground_data.pos = torch.from_numpy(
        ground_ligand.GetConformer().GetPositions()
    ).float()
    ground_data.edge_mask, ground_data.mask_rotate = TD_get_transformation_mask(
        ground_data
    )
    ground_data.edge_mask = torch.tensor(ground_data.edge_mask)
    G_ground = to_networkx(ground_data)

    try:
        data = featurize_mol(ligand)
    except RuntimeError as e:
        logging.error(f"Featurize ligand failed for complex {complex}")
        return

    data.edge_mask, data.mask_rotate = TD_get_transformation_mask(data)
    data.edge_mask = torch.tensor(data.edge_mask)
    G = to_networkx(data)

    ### TRANSLATIONAL STATE ###

    ground_ligand_center = torch.mean(ground_data.pos, dim=0, keepdim=True)
    glc = ground_ligand_center.squeeze().cpu().numpy()

    # store in CSV file

    with open(
        os.path.join(basedir, "sph" + ligand_name + ".csv"), "w", newline=""
    ) as csvf:
        writer = csv.writer(
            csvf, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([str(glc[0]), str(glc[1]), str(glc[2]), str(R), str(tr_gamma)])

    write_to_json(
        os.path.join(basedir, "sph" + ligand_name + ".json"),
        [str(glc[0]), str(glc[1]), str(glc[2]), str(R), str(tr_gamma)],
    )

    ### MATCHING ###

    GM = isomorphism.GraphMatcher(G, G_ground)
    # assert GM.is_isomorphic()
    if not GM.is_isomorphic():
        logging.error(f"Graphs are not isomorphic for complex {complex}")
        return

    mapping = np.array(list(GM.mapping.items()))
    match = -1
    for element in mapping:
        if element[0] == 0:
            match = element[1]
            break

    ### GET ROTATIONAL STATE OF THE GROUND LIGAND ###

    rot_state = get_rot_state(ground_data.pos)

    theta1, phi1 = get_rot_angles(rot_state[0])
    theta2, phi2 = get_rot_angles(rot_state[1])

    thM_1 = theta1 + theta_margin
    thm_1 = theta1 - theta_margin
    if thM_1 > math.pi:
        thM_1 = math.pi
    if thm_1 < 0:
        thm_1 = 0

    thM_2 = theta2 + theta_margin
    thm_2 = theta2 - theta_margin
    if thM_2 > math.pi:
        thM_2 = math.pi
    if thm_2 < 0:
        thm_2 = 0

    phM_1 = phi1 + phi_margin
    phm_1 = phi1 - phi_margin
    if phM_1 > 2 * math.pi:
        phM_1 = phM_1 - 2 * math.pi
    if phm_1 < 0:
        phm_1 = 2 * math.pi + phm_1

    phM_2 = phi2 + phi_margin
    phm_2 = phi2 - phi_margin
    if phM_2 > 2 * math.pi:
        phM_2 = phM_2 - 2 * math.pi
    if phm_2 < 0:
        phm_2 = 2 * math.pi + phm_2

    # # save to file
    rot_regions = [
        [thm_1, thM_1],
        [phm_1, phM_1],
        [thm_2, thM_2],
        [phm_2, phM_2],
    ]

    write_to_json(
        os.path.join(basedir, "rot" + ligand_name + ".json"),
        rot_regions,
    )

    ### ROTATABLE ANGLES MATCHING ###

    rotatable_bonds = []
    ground_rotatable_bonds = []

    for i, edge in enumerate(data.edge_index.T[data.edge_mask]):
        aux_bond = ligand.GetBondBetweenAtoms(int(edge[0]), int(edge[1]))
        rotatable_bonds.append(aux_bond.GetIdx())

        match1 = -1
        for element in mapping:
            if element[0] == int(edge[0]):
                match1 = int(element[1])
                break
        match2 = -1
        for element in mapping:
            if element[0] == int(edge[1]):
                match2 = int(element[1])
                break

        aux_ground_bond = ground_ligand.GetBondBetweenAtoms(match1, match2)
        ground_rotatable_bonds.append(aux_ground_bond.GetIdx())

    n_rotable_bonds = int(data.edge_mask.sum())

    if n_rotable_bonds > 0:
        assert len(ground_ligand.GetConformers()) == 1
        ordered_ground_rotatable_bonds = []

        for i, edge in enumerate(ground_data.edge_index.T[ground_data.edge_mask]):
            aux_bond = ground_ligand.GetBondBetweenAtoms(int(edge[0]), int(edge[1]))
            ordered_ground_rotatable_bonds.append(aux_bond.GetIdx())

        tau = TorsionCalculator.calc_torsion_angles(
            positions=ground_data.pos,
            edge_index=ground_data.edge_index,
            edge_mask=ground_data.edge_mask,
            mol=ground_data.mol,
            method=angle_calc_method,
        )

        rot_bond_interval = []

        ## NEW ALGORITHM, O(m) ##
        mx1 = max(ground_rotatable_bonds)
        mx2 = max(rotatable_bonds)

        if mx1 > mx2:
            max_edge_index = mx1
        else:
            max_edge_index = mx2

        aux_vec = [int(-1)] * (max_edge_index + 1)
        aux_aux_vec = [int(-1)] * (max_edge_index + 1)

        tau_ordered_as_rotatable_bonds = np.zeros(n_rotable_bonds)

        for i in range(n_rotable_bonds):
            aux_aux_vec[rotatable_bonds[i]] = i
        for i in range(n_rotable_bonds):
            aux_vec[ground_rotatable_bonds[i]] = rotatable_bonds[i]
        for i in range(n_rotable_bonds):
            tau_ordered_as_rotatable_bonds[
                aux_aux_vec[aux_vec[ordered_ground_rotatable_bonds[i]]]
            ] = tau[i]
        for i in range(n_rotable_bonds):

            if tau_ordered_as_rotatable_bonds[i] == math.e:
                value = "unable to define"
                rot_bond_interval.append([str(0), str(2 * math.pi)])

            else:
                value = str(tau_ordered_as_rotatable_bonds[i])
                tM = tau_ordered_as_rotatable_bonds[i] + tor_margin
                if tM > 2 * math.pi:
                    tM = tM - 2 * math.pi
                tm = tau_ordered_as_rotatable_bonds[i] - tor_margin
                if tm < 0:
                    tm = 2 * math.pi + tm
                rot_bond_interval.append([str(tm), str(tM)])

            if print_info:
                print(
                    "SMILES Bond "
                    + str(rotatable_bonds[i])
                    + " ---> Ground Bond "
                    + str(ground_rotatable_bonds[i])
                    + " = "
                    + value
                )

        write_to_json(
            os.path.join(basedir, "tor" + ligand_name + ".json"),
            rot_bond_interval,
        )

    return

@dataclass
class Ligand:
    name: str = None
    path: str = None
    smiles: str = None
    mol: Mol = None
    
def get_regions_from_template(
    protein_name: str,
    query_ligand: Ligand,
    template_ligand: Ligand,
    transfer_tr: bool,
    transfer_tor: bool,
    results_dir: str = None,
    create_files: bool = False,
    reorder_mols: bool = True,
):
    """
    Get regions from a template (a different docked ligand) instead of the ground truth
    
    if tr_file or tor_file are provided, they are used. if not, the regions for the
    template are computed
    """
    
    if transfer_tr:
        # look for the original file
        ...
        

def transfer_torsion_angles():
    ...