import torch
import ipdb
import copy
import os
import traceback
import random
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
import csv
from datasets.process_mols import write_mol_with_coords

from utils.diffusion_utils import (
    t_to_sigma as t_to_sigma_compl,
    get_t_schedule,
)
from utils.inference_utils import (
    InferenceDataset,
    set_nones,
    load_translation_data,
    load_rotation_data,
    load_torsion_data,
    update_radius,
    update_rot_boundaries,
    update_tor_boundaries,
)
from utils.torsion import AngleCalcMethod
from utils.guidance import GammaScheduler
from utils.parsing import parse_inference_args
from utils.sampling import randomize_position, sampling
from utils.utils import get_model, convert_string_to_float
from utils.pockets import get_pocket_center_p2rank, get_pocket_center_fpocket
from utils.visualise import PDBFile
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
import yaml

import warnings
warnings.filterwarnings("ignore") 

args = parse_inference_args()

os.makedirs(args.out_dir, exist_ok=True)
with open(f"{args.model_dir}/model_parameters.yml") as f:
    score_model_args = Namespace(**yaml.full_load(f))
if args.confidence_model_dir is not None:
    with open(f"{args.confidence_model_dir}/model_parameters.yml") as f:
        confidence_args = Namespace(**yaml.full_load(f))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.protein_ligand_csv is not None:
    df = pd.read_csv(args.protein_ligand_csv)
    complex_name_list_unproc = set_nones(df["complex_name"].tolist())
    complex_name_list = [
        name.replace(args.datadir, "", 1) for name in complex_name_list_unproc
    ]
    protein_path_list = set_nones(df["protein_path"].tolist())
    protein_sequence_list = set_nones(df["protein_sequence"].tolist())
    ligand_description_list = set_nones(df["ligand_description"].tolist())
else:
    complex_name_list = [args.complex_name]
    protein_path_list = [args.protein_path]
    protein_sequence_list = [args.protein_sequence]
    ligand_description_list = [args.ligand_description]

complex_name_list = [
    name if name is not None else f"complex_{i}"
    for i, name in enumerate(complex_name_list)
]


for name in complex_name_list:
    write_dir = f"{args.out_dir}/{name}"
    os.makedirs(write_dir, exist_ok=True)

# preprocess complexes into geomeric graphs
test_dataset = InferenceDataset(
    out_dir=args.out_dir,
    complex_names=complex_name_list,
    protein_files=protein_path_list,
    ligand_descriptions=ligand_description_list,
    protein_sequences=protein_sequence_list,
    lm_embeddings=score_model_args.esm_embeddings_path is not None,
    receptor_radius=score_model_args.receptor_radius,
    remove_hs=score_model_args.remove_hs,
    c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
    all_atoms=score_model_args.all_atoms,
    atom_radius=score_model_args.atom_radius,
    atom_max_neighbors=score_model_args.atom_max_neighbors,
    reorder_ligand=args.reordering,
    add_hs_conformer=args.add_hs_conformer,
    remove_hs_mol=args.remove_hs_mol,
)


test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

if (
    args.confidence_model_dir is not None
    and not confidence_args.use_original_model_cache
):
    print(
        "HAPPENING | confidence model uses different type of graphs than the score model. "
        "Loading (or creating if not existing) the data for the confidence model now."
    )
    confidence_test_dataset = InferenceDataset(
        out_dir=args.out_dir,
        complex_names=complex_name_list,
        protein_files=protein_path_list,
        ligand_descriptions=ligand_description_list,
        protein_sequences=protein_sequence_list,
        lm_embeddings=confidence_args.esm_embeddings_path is not None,
        receptor_radius=confidence_args.receptor_radius,
        remove_hs=confidence_args.remove_hs,
        c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
        all_atoms=confidence_args.all_atoms,
        atom_radius=confidence_args.atom_radius,
        atom_max_neighbors=confidence_args.atom_max_neighbors,
        precomputed_lm_embeddings=test_dataset.lm_embeddings,
        reorder_ligand=args.reordering,
        add_hs_conformer=args.add_hs_conformer,
        remove_hs_mol=args.remove_hs_mol,
    )
else:
    confidence_test_dataset = None


t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
state_dict = torch.load(
    f"{args.model_dir}/{args.ckpt}", map_location=torch.device("cpu")
)
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

if args.confidence_model_dir is not None:
    confidence_model = get_model(
        confidence_args,
        device,
        t_to_sigma=t_to_sigma,
        no_parallel=True,
        confidence_mode=True,
    )
    state_dict = torch.load(
        f"{args.confidence_model_dir}/{args.confidence_ckpt}",
        map_location=torch.device("cpu"),
    )
    confidence_model.load_state_dict(state_dict, strict=True)
    confidence_model = confidence_model.to(device)
    confidence_model.eval()
else:
    confidence_model = None
    confidence_args = None

tr_schedule = get_t_schedule(inference_steps=args.inference_steps)

failures, skipped = 0, 0
N = args.samples_per_complex
print("Size of test dataset: ", len(test_dataset))

for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
    complex_name = complex_name_list[idx]
    try:
        protein_name, ligand_name = complex_name.split('_')
    except ValueError: # ligand name isn't in the complex name, setting it as protein_name_ligand
        num = None
        try:  # for proteins with the same name + index (e.g. XXXX-1, XXXX-2, for two conformations of the protein XXXX that will use the same ligand)
            protein_name, num = complex_name.split('-')
            ligand_name = protein_name + '_ligand'
            
        except ValueError:
            protein_name = complex_name
            ligand_name = complex_name + '_ligand'

    # Load files.
    skip_translation = False
    if args.translation_guiding:
        if args.predict_pocket: # predicts pocket instead of taking one
            
            try:
                if args.predict_pocket_method == 'fpocket':
                    center = get_pocket_center_fpocket(
                        protein_path_list[idx]
                    )
                elif args.predict_pocket_method == 'p2rank':
                    center = get_pocket_center_p2rank(
                        protein_path_list[idx]
                    )
                else:
                    raise ValueError("Pocket prediction method not implemented")
                
                # add R and gamma and add dimension to support multiple
                # predicted pockets
                spheres = np.append(center, [7, 0.2]).reshape(1, -1)
            
            except Exception as e:
                print(e)
                skip_translation = True
            
        else:
            filename = os.path.join(
                args.datadir,
                complex_name,
                f"sph{ligand_name}.csv", 
            )
           
            spheres, skip_translation = load_translation_data(
                filename, 
                complex_name
            )
            
        if not skip_translation:
            # get true coordinates for the sphere after change in reference system
            try:
                current_center = torch.mean(orig_complex_graph["ligand"].pos, dim=0, keepdim=True)
            except:
                skipped += 1
                print(
                    f"HAPPENING | {name} has no coords so cant compute center. We are skipping this complex."
                )
                continue
            try:
                original_center: torch.Tensor = orig_complex_graph.original_center
            except:
                print(
                    f"HAPPENING | {name} has no original center something is wrong with the protein. We are skipping this complex."
                )
                continue
            
            # Correct sphere coordinates:
            # When doing predictions, DiffDock needs to center everything at (0,0,0). 
            # To apply this correction, we just substract the center of the protein,
            # saved in original_center
            for sphere in spheres:
                sphere[0:3] -= original_center.numpy()[0]

                if args.update_regions and args.tr_margin:
                    sphere = update_radius(sphere=sphere, new_radius=args.tr_margin)
    
    skip_rotation = False
    if args.rotation_guiding:
        filename = os.path.join(
            args.datadir,
            complex_name,
            f"rot{ligand_name}.json",
        )
        
        rotations, skip_rotation = load_rotation_data(
            filename,
            complex_name,
        )

        if not skip_rotation: 
            if args.update_regions and args.rot_margin:
                rotations = update_rot_boundaries(rotations, args.rot_margin)

    skip_torsion = False
    if args.torsion_guiding:
        filename = os.path.join(
            args.datadir,
            complex_name,
            f"tor{ligand_name}.json",
        )
            
        torsions, skip_torsion = load_torsion_data(
            filename, complex_name
        )

        if not skip_torsion:
            if not isinstance(torsions, np.ndarray):
                torsions = np.array(
                    [convert_string_to_float(item) for item in torsions], dtype=object
                )
            if args.n_regions == 1:
                torsions = torsions[:, :2] 

            if args.update_regions and args.tor_margin:
                torsions = update_tor_boundaries(torsions, args.tor_margin)

    if not orig_complex_graph.success[0]:
        skipped += 1
        print(
            f"HAPPENING | The test dataset did not contain {test_dataset.complex_names[idx]} for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}. We are skipping this complex."
        )
        continue
    try:
        if confidence_test_dataset is not None:
            confidence_complex_graph = confidence_test_dataset[idx]
            if not confidence_complex_graph.success:
                skipped += 1
                print(
                    f"HAPPENING | The confidence dataset did not contain {orig_complex_graph.name}. We are skipping this complex."
                )
                continue
            confidence_data_list = [
                copy.deepcopy(confidence_complex_graph) for _ in range(N)
            ]
        else:
            confidence_data_list = None
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
        randomize_position(
            data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max
        )
        lig = orig_complex_graph.mol[0]

        # initialize visualisation
        pdb = None
        if args.save_visualisation:
            visualization_list = []
            for graph in data_list:
                pdb = PDBFile(lig)
                pdb.add(lig, 0, 0)
                pdb.add(
                    (
                        orig_complex_graph["ligand"].pos
                        + orig_complex_graph.original_center
                    )
                    .detach()
                    .cpu(),
                    1,
                    0,
                )
                pdb.add(
                    (graph["ligand"].pos + graph.original_center).detach().cpu(),
                    part=1,
                    order=1,
                )
                visualization_list.append(pdb)
        else:
            visualization_list = None

        # run reverse diffusion
        n = args.actual_steps if args.actual_steps is not None else args.inference_steps

        if args.gamma_scheduler == 'onoff': # last_n
            tr_gamma_schedule = GammaScheduler.get_schedule(args.gamma_scheduler, n, last_n=5)
            rot_gamma_schedule = GammaScheduler.get_schedule(args.gamma_scheduler, n, last_n=5)
            tor_gamma_schedule = GammaScheduler.get_schedule(args.gamma_scheduler, n, last_n=5)
        elif args.gamma_scheduler == "warmup_cooldown":
            tr_gamma_schedule = GammaScheduler.get_schedule(args.gamma_scheduler, n, n_begin=3, n_end=3)
            rot_gamma_schedule = GammaScheduler.get_schedule(args.gamma_scheduler, n, n_begin=3, n_end=3)
            tor_gamma_schedule = GammaScheduler.get_schedule(args.gamma_scheduler, n, n_begin=3, n_end=3)
            
        else:
            tr_gamma_schedule = GammaScheduler.get_schedule(args.gamma_scheduler, n)
            rot_gamma_schedule = GammaScheduler.get_schedule(args.gamma_scheduler, n)
            tor_gamma_schedule = GammaScheduler.get_schedule(args.gamma_scheduler, n)
        
        data_list, confidence = sampling(
            data_list=data_list,
            model=model,
            inference_steps=(
                args.actual_steps
                if args.actual_steps is not None
                else args.inference_steps
            ),
            tr_schedule=tr_schedule,
            rot_schedule=tr_schedule,
            tor_schedule=tr_schedule,
            device=device,
            t_to_sigma=t_to_sigma,
            model_args=score_model_args,
            visualization_list=visualization_list,
            confidence_model=confidence_model,
            confidence_data_list=confidence_data_list,
            confidence_model_args=confidence_args,
            batch_size=args.batch_size,
            no_final_step_noise=args.no_final_step_noise,
            tr_guiding=args.translation_guiding if not skip_translation else False,
            desired_sphere=(
                spheres[0, :] # currently only works for 1 sphere
                if args.translation_guiding and not skip_translation
                else None
            ),
            tr_gamma_schedule=(
                tr_gamma_schedule
                if args.translation_guiding and not skip_translation
                else None
            ),
            rot_guiding=args.rotation_guiding if not skip_rotation else False,
            desired_rot=(
                rotations if args.rotation_guiding and not skip_rotation else None
            ),
            rot_gamma_schedule=(
                rot_gamma_schedule
                if args.rotation_guiding and not skip_rotation
                else None
            ),
            tor_guiding=args.torsion_guiding if not skip_torsion else False,
            desired_tor=torsions if args.torsion_guiding and not skip_torsion else None,
            tor_gamma_schedule=(
                tor_gamma_schedule
                if args.torsion_guiding and not skip_torsion
                else None
            ),
            dynamic_gamma=args.dynamic_gamma,
            angle_calc_method=args.angle_calc_method,
            Rm_update_method=args.Rm_update_method,
            reordering=args.reordering,
            neg_vdir=args.neg_vdir,
            save_inter_steps=args.save_inter_steps,
            results_dir=f"{args.out_dir}/{complex_name_list[idx]}",
            orig_complex_graph=orig_complex_graph,
            mask_n_tor=args.mask_n_tor if args.mask_n_tor else None,
            mask_n_distribution=(
                args.mask_n_distribution if args.mask_n_distribution else None
            ),
        )

        ligand_pos = np.asarray(
            [
                complex_graph["ligand"].pos.cpu().numpy()
                + orig_complex_graph.original_center.cpu().numpy()
                for complex_graph in data_list
            ]
        )

        # reorder predictions based on confidence output
        if confidence is not None and isinstance(
            confidence_args.rmsd_classification_cutoff, list
        ):
            confidence = confidence[:, 0]
        if confidence is not None:
            confidence = confidence.cpu().numpy()
            re_order = np.argsort(confidence)[::-1]
            confidence = confidence[re_order]
            ligand_pos = ligand_pos[re_order]

        # save predictions
        write_dir = f"{args.out_dir}/{complex_name_list[idx]}"
        
        for rank, pos in enumerate(ligand_pos):
            mol_pred = copy.deepcopy(lig)
            if score_model_args.remove_hs:
                mol_pred = RemoveHs(mol_pred)
            if rank == 0:
                write_mol_with_coords(
                    mol_pred, pos, os.path.join(write_dir, f"rank{rank+1}.sdf"),
                    properties = {
                        "rank": rank+1,
                        "confidence": confidence[rank],
                    },
                )
            write_mol_with_coords(
                mol_pred,
                pos,
                os.path.join(
                    write_dir, f"rank{rank+1}_confidence{confidence[rank]:.2f}.sdf",
                ),
                properties = {
                    "rank": rank+1,
                    "confidence": confidence[rank],
                },
            )

        # save visualisation frames
        if args.save_visualisation:
            if confidence is not None:
                for rank, batch_idx in enumerate(re_order):
                    visualization_list[batch_idx].write(
                        os.path.join(write_dir, f"rank{rank+1}_reverseprocess.pdb")
                    )
            else:
                for rank, batch_idx in enumerate(ligand_pos):
                    visualization_list[batch_idx].write(
                        os.path.join(write_dir, f"rank{rank+1}_reverseprocess.pdb")
                    )

    except Exception as e:
        traceback.print_exc()
        print("Failed on", orig_complex_graph["name"], e)
        failures += 1

print(f"Failed for {failures} complexes")
print(f"Skipped {skipped} complexes")
print(f"Results are in {args.out_dir}")
