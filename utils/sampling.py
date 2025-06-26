import numpy as np
import torch
import os
import copy

from torch_geometric.loader import DataLoader

from utils.diffusion_utils import modify_conformer, set_time
from utils.torsion import modify_conformer_torsion_angles, AngleCalcMethod
from scipy.spatial.transform import Rotation as R
from datasets.process_mols import write_mol_with_coords
from rdkit.Chem import RemoveHs
from utils.preprocessing_utils import mkdir_p


def randomize_position(data_list, no_torsion, no_random, tr_sigma_max):
    # in place modification of the list

    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(
                low=-np.pi, high=np.pi, size=complex_graph["ligand"].edge_mask.sum()
            )
            complex_graph["ligand"].pos = modify_conformer_torsion_angles(
                complex_graph["ligand"].pos,
                complex_graph["ligand", "ligand"].edge_index.T[
                    complex_graph["ligand"].edge_mask
                ],
                complex_graph["ligand"].mask_rotate[0],
                torsion_updates,
            )

    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph["ligand"].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        complex_graph["ligand"].pos = (
            complex_graph["ligand"].pos - molecule_center
        ) @ random_rotation.T
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph["ligand"].pos += tr_update


def sampling(
    data_list,
    model,
    inference_steps,
    tr_schedule,
    rot_schedule,
    tor_schedule,
    device,
    t_to_sigma,
    model_args,
    no_random=False,
    ode=False,
    visualization_list=None,
    confidence_model=None,
    confidence_data_list=None,
    confidence_model_args=None,
    no_final_step_noise=False,
    batch_size=32,
    tr_guiding=False,
    desired_sphere=None,
    tr_gamma_schedule=None,
    rot_guiding=False,
    desired_rot=None,
    rot_gamma_schedule=None,
    tor_guiding=False,
    desired_tor=None,
    tor_gamma_schedule=None,
    dynamic_gamma=False,
    angle_calc_method=AngleCalcMethod.TOR_CALC_2,
    Rm_update_method="m0",
    reordering=False,
    neg_vdir=True,
    save_inter_steps=False,
    results_dir=None,
    orig_complex_graph=None,
    mask_n_tor=None,
    mask_n_distribution=None,
):
    N = len(data_list)
    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = (
            tr_schedule[t_idx],
            rot_schedule[t_idx],
            tor_schedule[t_idx],
        )
        dt_tr = (
            tr_schedule[t_idx] - tr_schedule[t_idx + 1]
            if t_idx < inference_steps - 1
            else tr_schedule[t_idx]
        )
        dt_rot = (
            rot_schedule[t_idx] - rot_schedule[t_idx + 1]
            if t_idx < inference_steps - 1
            else rot_schedule[t_idx]
        )
        dt_tor = (
            tor_schedule[t_idx] - tor_schedule[t_idx + 1]
            if t_idx < inference_steps - 1
            else tor_schedule[t_idx]
        )

        tr_gamma = tr_gamma_schedule[t_idx] if tr_gamma_schedule else None
        rot_gamma = rot_gamma_schedule[t_idx] if rot_gamma_schedule else None
        tor_gamma = tor_gamma_schedule[t_idx] if tor_gamma_schedule else None

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
            set_time(
                complex_graph_batch, t_tr, t_rot, t_tor, b, model_args.all_atoms, device
            )

            with torch.no_grad():
                tr_score, rot_score, tor_score = model(complex_graph_batch)

            tr_g = tr_sigma * torch.sqrt(
                torch.tensor(
                    2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)
                )
            )
            rot_g = (
                2
                * rot_sigma
                * torch.sqrt(
                    torch.tensor(
                        np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)
                    )
                )
            )

            if ode:
                tr_perturb = (0.5 * tr_g**2 * dt_tr * tr_score.cpu()).cpu()
                rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g**2).cpu()
            else:
                tr_z = (
                    torch.zeros((b, 3))
                    if no_random
                    or (no_final_step_noise and t_idx == inference_steps - 1)
                    else torch.normal(mean=0, std=1, size=(b, 3))
                )
                tr_perturb = (
                    tr_g**2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z
                ).cpu()

                rot_z = (
                    torch.zeros((b, 3))
                    if no_random
                    or (no_final_step_noise and t_idx == inference_steps - 1)
                    else torch.normal(mean=0, std=1, size=(b, 3))
                )
                rot_perturb = (
                    rot_score.cpu() * dt_rot * rot_g**2
                    + rot_g * np.sqrt(dt_rot) * rot_z
                ).cpu()

            if not model_args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(
                    torch.tensor(
                        2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)
                    )
                )
                if ode:
                    tor_perturb = (0.5 * tor_g**2 * dt_tor * tor_score.cpu()).numpy()
                else:
                    tor_z = (
                        torch.zeros(tor_score.shape)
                        if no_random
                        or (no_final_step_noise and t_idx == inference_steps - 1)
                        else torch.normal(mean=0, std=1, size=tor_score.shape)
                    )
                    tor_perturb = (
                        tor_g**2 * dt_tor * tor_score.cpu()
                        + tor_g * np.sqrt(dt_tor) * tor_z
                    ).numpy()
                torsions_per_molecule = tor_perturb.shape[0] // b
            else:
                tor_perturb = None

            # denoising step
            new_data_list.extend(
                [
                    modify_conformer(
                        complex_graph,
                        tr_guiding,
                        tr_perturb[i : i + 1],
                        desired_sphere,
                        tr_gamma,
                        rot_guiding,
                        rot_perturb[i : i + 1].squeeze(0),
                        desired_rot,
                        rot_gamma,
                        tor_guiding,
                        (
                            tor_perturb[
                                i
                                * torsions_per_molecule : (i + 1)
                                * torsions_per_molecule
                            ]
                            if not model_args.no_torsion
                            else None
                        ),
                        desired_tor,
                        tor_gamma,
                        dynamic_gamma=dynamic_gamma,
                        angle_calc_method=angle_calc_method,
                        Rm_update_method=Rm_update_method,
                        reordering=reordering,
                        neg_vdir=neg_vdir,
                        mask_n_tor=mask_n_tor,
                        mask_n_distribution=mask_n_distribution,
                    )
                    for i, complex_graph in enumerate(
                        complex_graph_batch.to("cpu").to_data_list()
                    )
                ]
            )

        data_list = new_data_list

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add(
                    (data_list[idx]["ligand"].pos + data_list[idx].original_center)
                    .detach()
                    .cpu(),
                    part=1,
                    order=t_idx + 2,
                )

        if save_inter_steps and results_dir is not None:
            with torch.no_grad():
                if confidence_model is not None:
                    loader = DataLoader(data_list, batch_size=batch_size)
                    confidence_loader = iter(
                        DataLoader(confidence_data_list, batch_size=batch_size)
                    )
                    confidence = []
                    for complex_graph_batch in loader:
                        complex_graph_batch = complex_graph_batch.to(device)
                        if confidence_data_list is not None:
                            confidence_complex_graph_batch = next(confidence_loader).to(
                                device
                            )
                            confidence_complex_graph_batch["ligand"].pos = (
                                complex_graph_batch["ligand"].pos
                            )
                            set_time(
                                confidence_complex_graph_batch,
                                0,
                                0,
                                0,
                                N,
                                confidence_model_args.all_atoms,
                                device,
                            )
                            confidence.append(
                                confidence_model(confidence_complex_graph_batch)
                            )
                        else:
                            confidence.append(confidence_model(complex_graph_batch))
                    confidence = torch.cat(confidence, dim=0)
                else:
                    confidence = None

            lig = orig_complex_graph.mol[0]
            ligand_pos = np.asarray(
                [
                    complex_graph["ligand"].pos.cpu().numpy()
                    + orig_complex_graph.original_center.cpu().numpy()
                    for complex_graph in data_list
                ]
            )

            # reorder predictions based on confidence output
            if confidence is not None and isinstance(
                confidence_model_args.rmsd_classification_cutoff, list
            ):
                confidence = confidence[:, 0]
            if confidence is not None:
                confidence = confidence.cpu().numpy()
                re_order = np.argsort(confidence)[::-1]
                confidence = confidence[re_order]
                ligand_pos = ligand_pos[re_order]

            # save predictions
            write_dir = os.path.join(results_dir, f"step{t_idx}")
            mkdir_p(write_dir)  # create if doesnt exist
            for rank, pos in enumerate(ligand_pos):
                mol_pred = copy.deepcopy(lig)
                if model_args.remove_hs:
                    mol_pred = RemoveHs(mol_pred)
                if rank == 0:
                    write_mol_with_coords(
                        mol_pred, pos, os.path.join(write_dir, f"rank{rank+1}.sdf")
                    )
                write_mol_with_coords(
                    mol_pred,
                    pos,
                    os.path.join(
                        write_dir, f"rank{rank+1}_confidence{confidence[rank]:.2f}.sdf"
                    ),
                )

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(
                DataLoader(confidence_data_list, batch_size=batch_size)
            )
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch["ligand"].pos = complex_graph_batch[
                        "ligand"
                    ].pos
                    set_time(
                        confidence_complex_graph_batch,
                        0,
                        0,
                        0,
                        N,
                        confidence_model_args.all_atoms,
                        device,
                    )
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence
