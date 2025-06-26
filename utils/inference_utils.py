import os
import math
import copy
import torch
import json
import csv
import numpy as np
from Bio.PDB import PDBParser
from esm import FastaBatchedDataset, pretrained
from rdkit import Chem
from rdkit.Chem import AddHs, MolFromSmiles
from torch_geometric.data import Dataset, HeteroData
import esm
import ipdb

from datasets.process_mols import (
    parse_pdb_from_path,
    generate_conformer,
    read_molecule,
    get_lig_graph_with_matching,
    extract_receptor_structure,
    get_rec_graph,
)

from utils.torsion import AngleCalcMethod

three_to_one = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",  # MSE this is almost the same AA as MET. The sulfur is just replaced by Selen
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",
    "SER": "S",
    "SEC": "U",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "XAA": "X",
    "XLE": "J",
}
import optparse
from collections import defaultdict

def get_sequences_from_protfile(file_path):
    base, ext = os.path.splitext(file_path)
    if ext == '.pdb':
        return get_sequences_from_pdbfile(file_path)
    elif ext == '.mol2':
        return get_sequences_from_mol2file(file_path)
    
def get_sequences_from_mol2file(file_path):
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    chains = defaultdict(list)
    current_chain = None
    reading_atoms = False
    
    with open(file_path, 'r') as f:
        for line in f:
            if '@<TRIPOS>ATOM' in line:
                reading_atoms = True
                continue
            elif '@<TRIPOS>' in line:
                reading_atoms = False
                continue
                
            if reading_atoms and line.strip():
                parts = line.split()
                if len(parts) >= 8:
                    atom_name = parts[1]
                    res_name = parts[7][:3]  # Take first 3 characters of residue name
                    chain_id = parts[7][3] if len(parts[7]) > 3 else 'A'  # Default to chain A if no chain ID
                    res_num = parts[6]
                    
                    if atom_name == "CA":  # Only process alpha carbons
                        chains[chain_id].append((res_name, res_num))
    
    # Convert to sequence
    sequences = []
    for chain_id in sorted(chains.keys()):
        seq = ""
        prev_res_num = None
        
        for res_name, res_num in chains[chain_id]:
            # Avoid duplicate residues
            if res_num != prev_res_num:
                try:
                    if res_name in three_to_one:
                        seq += three_to_one[res_name]
                    else:
                        seq += "-"
                        print(f"encountered unknown AA: {res_name} in the complex. Replacing it with a dash - .")
                except Exception as e:
                    seq += "-"
                    print(f"Error processing residue {res_name}: {str(e)}")
                prev_res_num = res_num
    
        sequences.append(seq)
    
    return ":".join(sequences)

def get_sequences_from_pdbfile(file_path):
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure("random_id", file_path)
    structure = structure[0]
    sequence = None
    for i, chain in enumerate(structure):
        seq = ""
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
            if (
                c_alpha != None and n != None and c != None
            ):  # only append residue if it is an amino acid
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += "-"
                    print(
                        "encountered unknown AA: ",
                        residue.get_resname(),
                        " in the complex. Replacing it with a dash - .",
                    )

        if sequence is None:
            sequence = seq
        else:
            sequence += ":" + seq

    return sequence


def set_nones(l):
    return [s if str(s) != "nan" else None for s in l]


def get_sequences(protein_files, protein_sequences):
    new_sequences = []
    for i in range(len(protein_files)):
        if protein_files[i] is not None:
            new_sequences.append(get_sequences_from_protfile(protein_files[i]))
        else:
            new_sequences.append(protein_sequences[i])
            
    return new_sequences


def compute_ESM_embeddings(model, alphabet, labels, sequences):
    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    include = "per_tok"
    truncation_seq_length = 1022

    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers
    ]
    embeddings = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1 : truncate_len + 1].clone()
    return embeddings


def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory on chunk_size", chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None


class InferenceDataset(Dataset):
    def __init__(
        self,
        out_dir,
        complex_names,
        protein_files,
        ligand_descriptions,
        protein_sequences,
        lm_embeddings,
        receptor_radius=30,
        c_alpha_max_neighbors=None,
        precomputed_lm_embeddings=None,
        remove_hs=False,
        all_atoms=False,
        atom_radius=5,
        atom_max_neighbors=None,
        reorder_ligand=False,
        add_hs_conformer=True,
        remove_hs_mol=False,
    ):

        super(InferenceDataset, self).__init__()
        self.receptor_radius = receptor_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors

        self.complex_names = complex_names
        self.protein_files = protein_files
        self.ligand_descriptions = ligand_descriptions
        self.protein_sequences = protein_sequences

        self.reorder_ligand = reorder_ligand
        self.add_hs_conformer = add_hs_conformer
        self.remove_hs_mol = remove_hs_mol

        # generate LM embeddings
        if lm_embeddings and (
            precomputed_lm_embeddings is None or precomputed_lm_embeddings[0] is None
        ):
            print("Generating ESM language model embeddings")
            model_location = "esm2_t33_650M_UR50D"
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            protein_sequences = get_sequences(protein_files, protein_sequences)
            labels, sequences = [], []
            for i in range(len(protein_sequences)):
                s = protein_sequences[i].split(":")
                sequences.extend(s)
                labels.extend(
                    [complex_names[i] + "_chain_" + str(j) for j in range(len(s))]
                )

            lm_embeddings = compute_ESM_embeddings(model, alphabet, labels, sequences)

            self.lm_embeddings = []
            for i in range(len(protein_sequences)):
                s = protein_sequences[i].split(":")
                self.lm_embeddings.append(
                    [
                        lm_embeddings[f"{complex_names[i]}_chain_{j}"]
                        for j in range(len(s))
                    ]
                )

        elif not lm_embeddings:
            self.lm_embeddings = [None] * len(self.complex_names)

        else:
            self.lm_embeddings = precomputed_lm_embeddings

        # generate structures with ESMFold
        if None in protein_files:
            print("generating missing structures with ESMFold")
            model = esm.pretrained.esmfold_v1()
            model = model.eval().cuda()

            for i in range(len(protein_files)):
                if protein_files[i] is None:
                    self.protein_files[i] = (
                        f"{out_dir}/{complex_names[i]}/{complex_names[i]}_esmfold.pdb"
                    )
                    if not os.path.exists(self.protein_files[i]):
                        print("generating", self.protein_files[i])
                        generate_ESM_structure(
                            model, self.protein_files[i], protein_sequences[i]
                        )

    def len(self):
        return len(self.complex_names)

    def get(self, idx):

        name, protein_file, ligand_description, lm_embedding = (
            self.complex_names[idx],
            self.protein_files[idx],
            self.ligand_descriptions[idx],
            self.lm_embeddings[idx],
        )

        # build the pytorch geometric heterogeneous graph
        complex_graph = HeteroData()
        complex_graph["name"] = name

        # parse the ligand, either from file or smile
        try:
            is_path = ligand_description.endswith('sdf')
            if not is_path:
                mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path
                if self.reorder_ligand:
                    mol_ = copy.deepcopy(mol)
                    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol_))
                if mol is not None:
                    if self.add_hs_conformer:
                        mol = AddHs(mol)
                    generate_conformer(mol)
            else:
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if mol is None:
                    raise Exception(
                        "RDKit could not read the molecule ", ligand_description
                    )
                if self.reorder_ligand:
                    mol_ = copy.deepcopy(mol)
                    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol_))
                if mol is not None:
                    if self.add_hs_conformer:
                        mol = AddHs(mol)
                    generate_conformer(mol)

        except Exception as e:
            print(
                "Failed to read molecule ",
                ligand_description,
                " We are skipping it. The reason is the exception: ",
                e,
            )
            complex_graph["success"] = False
            return complex_graph

        try:
            # parse the receptor from the pdb file
            rec_model = parse_pdb_from_path(protein_file)
            get_lig_graph_with_matching(
                mol,
                complex_graph,
                popsize=None,
                maxiter=None,
                matching=False,
                keep_original=False,
                num_conformers=1,
                remove_hs=self.remove_hs,
            )
            rec, rec_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = (
                extract_receptor_structure(
                    rec_model, mol, lm_embedding_chains=lm_embedding
                )
            )
            if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                print(
                    f"LM embeddings for complex {name} did not have the right length for the protein. Skipping {name}."
                )
                complex_graph["success"] = False
                return complex_graph

            get_rec_graph(
                rec,
                rec_coords,
                c_alpha_coords,
                n_coords,
                c_coords,
                complex_graph,
                rec_radius=self.receptor_radius,
                c_alpha_max_neighbors=self.c_alpha_max_neighbors,
                all_atoms=self.all_atoms,
                atom_radius=self.atom_radius,
                atom_max_neighbors=self.atom_max_neighbors,
                remove_hs=self.remove_hs,
                lm_embeddings=lm_embeddings,
            )

        except Exception as e:
            print(f"Skipping {name} because of the error:")
            print(e)
            complex_graph["success"] = False
            return complex_graph

        # getting the original ligand center is necessary because the complex is
        # centered at 0.(which. For translational
        # guidance, this means that the ground truth will need to be moved as well
        original_ligand_center = torch.mean(
            complex_graph["ligand"].orig_pos, dim=0, keepdim=True
        )
        protein_center = torch.mean(complex_graph["receptor"].pos, dim=0, keepdim=True)
        complex_graph["receptor"].pos -= protein_center
        if self.all_atoms:
            complex_graph["atom"].pos -= protein_center

        ligand_center = torch.mean(complex_graph["ligand"].pos, dim=0, keepdim=True)
        complex_graph["ligand"].pos -= ligand_center

        complex_graph.original_center = protein_center
        complex_graph.original_ligand_center = original_ligand_center
        if self.remove_hs_mol:
            mol = Chem.RemoveHs(mol)
        complex_graph.mol = mol
        complex_graph["success"] = True
        return complex_graph

# regions

def update_radius(
    sphere: np.ndarray,
    new_radius: float
):
    """Sets the value of the radius used to build the sphere.

    For the sphere, we can just set the value that we want.
    Parameters
    ----------
    initial_radius : float
        initial radius used
    new_radius : float
        new radius
    """
    sphere[3] = new_radius
    return sphere

def update_rot_boundaries(
    rotations: np.ndarray,
    eta: float = 0.15,
):
    """
    Updates the values of the regions for rotation.
    
    It should be used strictly to update the boundaries of some regions
    previously defined with some eta.
    
    For rotation, the regions are defined by a given \eta, and therefore
    we can't just set the values.

    Parameters
    ----------
    rotations : np.ndarray
        original regions
    eta : float, optional
        rotation margin, by default 0.15
    """
    def _adjust_range(range_vals, max_val, delta):
        lower_bound, upper_bound = range_vals
        min_val = (lower_bound - delta) % max_val
        max_val = (upper_bound + delta) % max_val
        return np.array([min_val, max_val])
    
    a, b = rotations[0][:2]
    if a > b:
        a, b = b, a
    
    original_eta = round((b - a) / 2, 2)
    delta_eta = eta - original_eta
    
    theta_max = np.pi # theta values are in [0, pi)
    phi_max = 2 * np.pi # phi valuesa re in [0, 2pi)
    
    updated_rotations = []
    for i in range(0, len(rotations), 2):
        theta_range = _adjust_range(rotations[i], theta_max, delta_eta)
        phi_range = _adjust_range(rotations[i + 1], phi_max, delta_eta)
        updated_rotations.extend([theta_range, phi_range])

    return np.array(updated_rotations)
       
def update_tor_boundaries(
    torsions: np.ndarray,
    eta: float = 0.15,
    tolerance: float = 1e-10,
):
    """
    Updates the values of the regions for torsion.
    
    It should be used strictly to update the boundaries of some regions
    previously defined with some eta.
    
    For torsions, the regions are defined by a given \eta, and therefore
    we can't just set the values.

    Parameters
    ----------
    torsions : np.ndarray
        Current values for torsion regions
    eta :
        torsion margin (used to build the torsion region with value +- eta), by default 0.15
    tolerance : float, optional
         by default 1e-10
    """
    
    wrap_angle = lambda angle: round(angle % (2 * math.pi), 3)
    is_near_0_or_2pi = lambda angle: np.isclose(angle, 0, atol=tolerance) or np.isclose(angle, 2 * np.pi, atol=tolerance)
    
    first_region = torsions[0][:2] # first region
    
    # obtain the original eta
    l1 = abs(first_region[0] - first_region[1]) # distance along one way of the circle
    l2 = 2 * np.pi - l1 # distance along the other way of the circle
    dist = min(l1, l2) # shortest distance
    
    original_eta = round(dist / 2, 2)
    
    delta_eta = eta - original_eta  
    
    for i in range(torsions.shape[1]):
        for j in range(torsions.shape[0] - 1):
            if not is_near_0_or_2pi(torsions[j, i]):
                if i % 2 == 0:  # Subtract delta_eta for even-indexed columns
                    torsions[j, i] = wrap_angle(torsions[j, i] - delta_eta)
                else:  # Add delta_eta for odd-indexed columns
                    torsions[j, i] = wrap_angle(torsions[j, i] + delta_eta)
    
    return torsions


def load_translation_data(
    filename: str, 
    name: str
):
    skip_translation = False

    if not os.path.exists(filename):
        skip_translation = True
        print(f"Couldn't find {filename}. Skipping translation for {str(name)}")
        return None, skip_translation

    spheres = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row_idx, row in enumerate(reader):
            spheres.append([float(i) for i in row])

    spheres = np.asarray(spheres)
    return spheres, skip_translation

def load_rotation_data(
    filename: str, 
    name: str,
):
    skip_rotation = False

    if not os.path.exists(filename):
        skip_rotation = True
        print(f"Couldn't find {filename}. Skipping rotation for {name}")
        return None, skip_rotation

    with open(filename, "r") as file:
        rotations_list = json.load(file)

    if not rotations_list:  # Check if the list is empty
        skip_rotation = True
        print(f"No rotation data found. Skipping rotation for {name}")
        return None, skip_rotation
    
    rotations = np.array(rotations_list, dtype=float)
    return rotations, skip_rotation

def load_torsion_data(
    filename: str, 
    name: str,
):
    skip_torsion = False
    if not os.path.exists(filename):
        skip_torsion = True
        print(f"Couldn't find {filename}. Skipping torsion for {str(name)}")
        return None, skip_torsion

    with open(filename, "r") as file:
        torsions_list = json.load(file)

    if torsions_list is None:
        skip_torsion = print(f"No torsion data found. Skipping rotation for {name}")
        return None, skip_torsion
    
    torsions = np.array(torsions_list, dtype=float)
    return torsions, skip_torsion
