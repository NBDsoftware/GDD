import os
import csv
import logging
import subprocess
import time
import shutil
import pandas as pd
import argparse
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from urllib.request import urlopen
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit.Chem import rdmolfiles
from rdkit import Chem
from utils.preprocessing_utils import (
    mkdir_p,
    separate_pdb_ligands,
    process_obabel,
    select_protein_chains,
    get_ground_pos_func,
    NoChainsWithinCutoff,
    get_regions_from_template,
    Ligand
)
from utils.inference_utils import (
    load_translation_data,
    load_torsion_data
)
import ipdb
from tqdm import tqdm
from utils.torsion import AngleCalcMethod


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def angle_calc_method_type(string):
    try:
        return AngleCalcMethod[string.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid value for AngleCalcMethod: {string}")


parser = ArgumentParser()
parser.add_argument(
    "--data_directory",
    type=str,
    required=False,
    default=None,
    help="Path to the directory with the pdb files of the complexes",
)
parser.add_argument(
    "--structured_directory",
    action="store_true",
    help="Input directory has complex subfolders and separated ligand files",
)
parser.add_argument(
    "--skip_protein_processing",
    action="store_true",
    help="Skip protein processing steps (cleaning, chain selection)",
)
parser.add_argument(
    "--in_place",
    action="store_true",
    help="Process files in the input directory instead of creating a new _processed directory",
)

parser.add_argument(
    "--get_ground_ligand_pose",
    action="store_true",
    help="whether or not it gets the ligand pose. Not getting can be useful if it was obtained \
                        somewhere else, for example with angle transfers for torsion.",
)

parser.add_argument(
    "--processed_data_csv",
    default=None,
    action='store_true',
    required=False,
)
parser.add_argument(
    "--get_regions_from_template",
    type=bool,
    default=False,
    help="Uses a template molecule to get regions"
)

parser.add_argument(
    "--smiles_source",
    type=str, # mol, rcsb
    default="mol",
    required=False,
)
parser.add_argument(
    "--ligand_ext", 
    type=str,
    default="sdf",
    required=False,
)


def main():
    args = parser.parse_args()
    
    # Handle directory setup
    args.data_directory = args.data_directory.rstrip("/")
    if args.in_place:
        results_directory = args.data_directory
    else:
        results_directory = args.data_directory + "_processed"
        mkdir_p(results_directory)
        
        # If directory is structured but not in_place, copy over the structure
        if args.structured_directory:
            for complex_name in os.listdir(args.data_directory):
                src_path = os.path.join(args.data_directory, complex_name)
                if os.path.isdir(src_path):
                    dst_path = os.path.join(results_directory, complex_name)
                    mkdir_p(dst_path)
                    # Copy necessary files (like ligands)
                    for file in os.listdir(src_path):
                        if file.endswith(('.pdb', '.sdf', '.mol2')):  # Add other extensions if needed
                            shutil.copy2(
                                os.path.join(src_path, file),
                                os.path.join(dst_path, file)
                            )
        
    logging.basicConfig(
        filename=os.path.join(results_directory, "processing_errors.log"),
        level=logging.ERROR,
    )
    if not args.structured_directory:

        # Process raw PDB files into complex directories
        for pdb_file in os.listdir(args.data_directory):
            protein_name = pdb_file.strip(".pdb")
            pdb_file_fullpath = os.path.join(args.data_directory, pdb_file)

            ### 0. CREATE DIRECTORY ###
            pdb_dir = os.path.join(results_directory, protein_name)
            mkdir_p(pdb_dir)

            ### 1. SEPARATE PROTEIN AND LIGANDS ###
            try:
                separate_pdb_ligands(
                    pdb_file=pdb_file_fullpath, 
                    results_path=results_directory,
                )
            except:
                print(f"Failed separating pdb ligands with {complex_name}")
                continue
    
    # At this point we have a directory of complexes regardless of the original
    # dataset structure
    csv_file = os.path.join(results_directory, "data.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as csvf:
            spamwriter = csv.writer(
                csvf, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            spamwriter.writerow(
                [
                    "complex_name",
                    "protein_path",
                    "ligand_description",
                    "protein_sequence",
                ]
            )
            
        for complex_name in os.listdir(results_directory):
            if not os.path.isdir(os.path.join(results_directory, complex_name)):
                continue
            
            try:
                protein_name, ligand_name = complex_name.split('_')
            except ValueError:
                
                num = None
                try:  # for proteins with the same name + index:
                    protein_name, num = complex_name.split('-')
                    ligand_name = protein_name + '_ligand'
                    
                except ValueError:
                    protein_name = complex_name
                    ligand_name = complex_name + '_ligand'
            
            complex_path = os.path.join(results_directory, complex_name)
            ligand_ext = args.ligand_ext.strip(".")
            ligand_path = os.path.join(complex_path, f"{ligand_name}.{ligand_ext}")
            if num is not None:
                protein_path = os.path.join(complex_path, f"{protein_name}_protein.pdb")
                processed_protein_path = os.path.join(complex_path, f"{protein_name}_protein_processed.pdb")
            else:
                protein_path = os.path.join(complex_path, f"{protein_name}-{num}.pdb")
                processed_protein_path = os.path.join(complex_path, f"{protein_name}-{num}_processed.pdb")
            
            if not args.skip_protein_processing:                
                ### 1. SEPARATE PROTEIN AND LIGANDS ###
                # We already have the ligand files, so this part is to remove the rest of the 
                # ligands from the protein. We do so by providing the path of the 
                # ligand.
                try:
                    separate_pdb_ligands(
                        pdb_file=protein_path, 
                        results_path=results_directory,
                        complex_name=complex_name,
                        ligand_file=ligand_path
                    )
                except:
                    print(f"Failed separating pdb ligands with {complex_name}")
                    continue
            
                ### 2. PROCESS WITH OBABEL
                try:
                    process_obabel(protein_path, remove=True)
                except:
                    print(f"Failed processing with obabel with {complex_name}")
                    continue
                
                ### 3. REMOVE CHAINS THAT AREN'T WITHIN 10A OF THE LIGAND ###
                try:
                    processed_protein_path = select_protein_chains(
                        protein_name,
                        protein_path, 
                        ligand_path, 
                        cutoff=10
                    )
                except NoChainsWithinCutoff:
                    print(
                        "When selecting chains, no chains within 10 A angstroms of the ligand were found"
                    )
                    logging.error(
                        f"NoChainsWithinCutoff: No chains within the cutoff distance to the ligand for complex {complex_name}"
                    )
                    continue
                except:
                    print("Something else failed when selecting the chains")
                    logging.error(
                        f"UnexpectedChainError: Unexpected error in select_protein_chains for complex {complex_name}"
                    )
                    continue

            ### 4. GET LIGAND_GROUND_POS AND CREATE THE CSV FILES ###
            if args.get_ground_ligand_pose:
                if args.smiles_source =='mol':
                    if ligand_ext == 'pdb':
                        ligand_mol = Chem.MolFromPDBFile(ligand_path)
                    elif ligand_ext == 'sdf':
                        ligand_mol = Chem.MolFromMolFile(ligand_path)
                        
                    ligand_smiles = Chem.MolToSmiles(ligand_mol)
                
                else:
                    raise ValueError("bad smiles source")
                
                try:
                    get_ground_pos_func(
                        protein_name=protein_name,
                        ligand_name=ligand_name,
                        ligand_smiles=ligand_smiles,
                        ligand_file=ligand_path,
                        create_files=True,
                        results_dir=results_directory
                    )
                except:
                    print(f"Failed getting ground truth for {complex_name}")
                    continue

            # Write to CSV
            with open(csv_file, "a", newline="") as csvf:
                writer = csv.writer(
                    csvf, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
                )
                writer.writerow([
                    complex_name,
                    processed_protein_path if not args.skip_protein_processing else protein_path,
                    ligand_smiles or "",
                    ""
                ])
           
    else: 
        raise NotImplementedError(
            f"data.csv already exists. Processing in this setting not implemented yet, delete the data.csv file and retry"
        )           
                


if __name__ == "__main__":
    main()
