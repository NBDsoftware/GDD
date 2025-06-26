import os
import csv
import json
import re
import torch
import ipdb
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

from argparse import FileType, ArgumentParser

from datasets.process_mols import read_molecule, read_sdf_or_mol2
from utils.featurization import featurize_mol
from utils.preprocessing_utils import mkdir_p, reorder_mol, TD_get_transformation_mask
from utils.inference_utils import set_nones
from utils.metrics import MetricsCalculator, Pose


parser = ArgumentParser()
parser.add_argument(
    "--protein_ligand_csv",
    type=str,
    default="data/6drg.csv",
    help="Path to a .csv file specifying the input as described in the README",
)
parser.add_argument(
    "--datadir",
    default="data/",
    help="Path to the folder with the data proceesed by process_data.py",
)
parser.add_argument(
    "--results_path", type=str, help="Path to folder with results (.sdf files)"
)
parser.add_argument(
    "--num_predictions",
    type=int,
    default=40,
    help="Number of generated conformers/poses",
)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument(
    "--ligand_ext", 
    type=str,
    default="sdf",
    required=False,
)
parser.add_argument(
    "--save_readings",
    action="store_true",
    default=False,
    help="Whether to save results to txt file",
)
parser.add_argument(
    "--print_info",
    action="store_true",
    default=False,
    help="Whether to print minimum, maximum and mean RMSD for each complex",
)

parser.add_argument("--top_hits_only", default=False)
parser.add_argument("--inter_steps", action="store_true")
args = parser.parse_args()

df = pd.read_csv(args.protein_ligand_csv)
complex_name_list = set_nones(df["complex_name"].tolist())
complex_name_list = [name.replace(args.datadir, "", 1) for name in complex_name_list]
protein_path_list = set_nones(df["protein_path"].tolist())
ligand_description_list = set_nones(df["ligand_description"].tolist())
protein_sequence_list = set_nones(df["protein_sequence"].tolist())

metrics_results_path = args.results_path.replace("results/", "metrics/")
mkdir_p(metrics_results_path)


print(f"{len(complex_name_list)=}")
for idx, complex_name in enumerate(complex_name_list):

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

    ext = args.ligand_ext.strip(".")
    # load smiles ligand
    try:
        ligand = Chem.MolFromSmiles(ligand_description_list[idx])
        ligand = reorder_mol(ligand)

    except Exception as e:
        print(f"Couldn't load ligand from smiles. Skipping")
        continue

    print(f"Processing complex {complex_name}")
    try:
        ground_ligand = read_molecule(
            os.path.join(args.datadir, complex_name, f'{ligand_name}.{ext}'),
            sanitize=True,
            calc_charges=False,
            remove_hs=True,
        )

        ground_ligand = reorder_mol(ground_ligand)

    except Exception as e:
        print(f"Something went wrong loading ground ligand, {e}, skipping")
        continue

    try:
        ground_ligand = AllChem.AssignBondOrdersFromTemplate(ligand, ground_ligand)
    except:
        print(f"Couldn't assign bond orders, skipping")
        continue

    # load predicted mols and sort by confidence
    if args.inter_steps:
        for step_i in range(args.steps):
            predicted_mols = []

            step_dir = os.path.join(args.results_path, complex_name, f"step{step_i}")
            if not os.path.exists(step_dir):
                continue

            filepaths = sorted(os.listdir(step_dir))
            sorted_filenames = sorted(
                filter(lambda x: x != "rank1.sdf", filepaths),
                key=lambda x: (
                    int(re.search(r"rank(\d+)", x).group(1))
                    if re.search(r"rank(\d+)", x)
                    else float("inf")
                ),
            )

            for i, filepath in enumerate(sorted_filenames):
                try:
                    predicted_mol = Chem.MolFromMolFile(
                        os.path.join(step_dir, filepath)
                    )
                    predicted_mol = reorder_mol(predicted_mol)
                    predicted_mols.append(predicted_mol)
                except:
                    print(f"Couldn't read mol for prediction {i}")

            # turn mols into Poses
            try:
                ground_data = featurize_mol(ground_ligand)
                ground_data.mol = ground_ligand
                ground_data.edge_mask, ground_data.mask_rotate = TD_get_transformation_mask(
                    ground_data
                )
                ground_data.edge_mask = torch.tensor(ground_data.edge_mask)

                edge_index = ground_data.edge_index
                edge_mask = ground_data.edge_mask

            except:
                print(f"Couldn't featurize ground ligand for {complex_name}. Skipping...")
                continue
                
            ground_pose = Pose(
                mol=ground_ligand,
                pos=ground_ligand.GetConformer().GetPositions(),
                edge_index=edge_index,
                edge_mask=edge_mask,
            )

            predicted_poses = []
            for mol in predicted_mols:
                try:
                    pose = Pose(
                        mol=mol,
                        pos=mol.GetConformer().GetPositions(),
                        edge_index=edge_index,
                        edge_mask=edge_mask,
                    )
                    predicted_poses.append(pose)
                except:
                    print(f"Couldn't get pose {i} for {complex_name}. Skipping...")

            # Compute metrics
            try:
                metrics = MetricsCalculator.compute_all_metrics(
                    ground_pose, predicted_poses
                )
            except:
                print(f"Couldn't compute metrics for ligand {complex_name}. Skipping...")
                continue

            # Save results
            MetricsCalculator.save_metrics(
                metrics=metrics,
                path=os.path.join(
                    metrics_results_path, complex_name, f"step{step_i}" + ".json"
                ),
            )
    
    else:
        predicted_mols = []
        mol_results_path = os.path.join(args.results_path, complex_name)
                
        if not os.path.exists(mol_results_path): # failed for this mol, skip
            print(f"skipping {complex_name} @ {mol_results_path}")
            continue
        
        pred_mols_paths = sorted(os.listdir(mol_results_path))
        sorted_pred_mols_paths = sorted(
            filter(lambda x: x != "rank1.sdf", pred_mols_paths),
            key=lambda x: (
                int(re.search(r"rank(\d+)", x).group(1))
                if re.search(r"rank(\d+)", x)
                else float("inf")
            ),
        )
                
        for i, mol in enumerate(sorted_pred_mols_paths):
            if "reverseprocess" in mol:
                continue
            try:
                predicted_mol = Chem.MolFromMolFile(
                    os.path.join(mol_results_path, mol)
                ) 
                predicted_mol = reorder_mol(predicted_mol)
                predicted_mols.append(predicted_mol)
            except:
                print(f"Couldn't read mol for prediction {i}")
                
        # turn into poses
        try:
            ground_data = featurize_mol(ground_ligand)
            ground_data.mol = ground_ligand
            ground_data.edge_mask, ground_data.mask_rotate = TD_get_transformation_mask(ground_data)
            ground_data.edge_mask = torch.tensor(ground_data.edge_mask)
        except:
            print(f"Couldn't featurize ground ligand for {complex_name}. Skipping...")
            continue

        edge_index = ground_data.edge_index
        edge_mask = ground_data.edge_mask
        
        ground_pose = Pose(
            mol=ground_ligand,
            pos=ground_ligand.GetConformer().GetPositions(),
            edge_index=edge_index,
            edge_mask=edge_mask,
        )
                
        predicted_poses = []
        for i, mol in enumerate(predicted_mols):
            try:
                pose = Pose(
                    mol=mol,
                    pos=mol.GetConformer().GetPositions(),
                    edge_index=edge_index,
                    edge_mask=edge_mask,
                )
            except:
                print(f"Couldn't get pose {i} for {complex_name}. Skipping...")
                
            predicted_poses.append(pose)
        
        # Compute metrics
        try:
            metrics = MetricsCalculator.compute_all_metrics(ground_pose, predicted_poses)
        except:
            print(f"Couldn't compute metrics for ligand {complex_name}. Skipping...")
            continue
                
        # Save results
        MetricsCalculator.save_metrics(
            metrics=metrics,
            path=os.path.join(metrics_results_path, complex_name + '.json')
        )
        