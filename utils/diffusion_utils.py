import math
import numpy as np
import torch
import torch.nn.functional as F
import random

from torch import nn

from utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from utils.torsion import modify_conformer_torsion_angles, AngleCalcMethod
from utils.guidance import (
    tr_guider,
    in_tr_region,
    compute_tr_gamma,
    get_guided_tr_update,
    get_rot_state,
    rot_guider,
    in_rot_region,
    get_guided_rotation_matrix,
    get_tor_state,
    tor_guider,
    in_torus_region,
    compute_tor_gamma,
    get_guided_tor_update,
)


def t_to_sigma(t_tr, t_rot, t_tor, args):
    tr_sigma = args.tr_sigma_min ** (1 - t_tr) * args.tr_sigma_max**t_tr
    rot_sigma = args.rot_sigma_min ** (1 - t_rot) * args.rot_sigma_max**t_rot
    tor_sigma = args.tor_sigma_min ** (1 - t_tor) * args.tor_sigma_max**t_tor
    return tr_sigma, rot_sigma, tor_sigma


def modify_conformer(
    data,
    tr_guiding,
    tr_update,
    desired_tr,
    tr_gamma,
    rot_guiding,
    rot_update,
    desired_rot,
    rot_gamma,
    tor_guiding,
    tor_updates,
    desired_tor,
    tor_gamma,
    dynamic_gamma: bool = False,
    angle_calc_method=AngleCalcMethod.TOR_CALC_2,
    Rm_update_method="m0",
    reordering=True,
    neg_vdir=True,
    mask_n_tor=None,
    mask_n_distribution="random",
):

    ### GET CURRENT TRANSLATION, ROTATION AND TORSIONAL STATES ###
    ligand_center = torch.mean(data["ligand"].pos, dim=0, keepdim=True)
    if tr_guiding:
        tr_state = ligand_center
    if rot_guiding:
        rot_state = get_rot_state(data["ligand"].pos)
    if tor_guiding:
        tor_state = get_tor_state(
            positions=data["ligand"].pos,
            edge_index=data["ligand", "ligand"].edge_index,
            edge_mask=data["ligand"].edge_mask,
            mol=data.mol[0],
            method=angle_calc_method,
        )

    ########################## TRANSLATIONAL GUIDING ##########################
    # --------------------------------------------------------------------------#

    if tr_guiding and not in_tr_region(tr_state, desired_tr):
        tr_vdir, tr_distance = tr_guider(
            pos=tr_state.cpu().numpy(), sph=desired_tr
        )  # vdir: direction in which to guide the ligand
        tr_update = tr_update.cpu().numpy()

        if dynamic_gamma:
            tr_gamma, tr_sim = compute_tr_gamma(
                protein_diameter=data["receptor"].diameter,
                tr_update=tr_update,
                vdir=tr_vdir,
                distance=tr_distance,
            )

        tr_update = get_guided_tr_update(
            current_state=tr_state,
            vdir=tr_vdir,
            distance=tr_distance,
            update=tr_update,
            gamma=tr_gamma,
            update_method=Rm_update_method,
        )
        tr_update = torch.from_numpy(tr_update.astype(np.float32))

    ############################ ROTATIONAL GUIDING ############################
    # --------------------------------------------------------------------------#

    if rot_guiding and not in_rot_region(state=rot_state, region=desired_rot):
        rot_vdir, rot_distance = rot_guider(rot_state, desired_rot)
        rot_mat = get_guided_rotation_matrix(
            current_state=rot_state,
            vdir=rot_vdir,
            distance=rot_distance,
            rot_update=rot_update,
            gamma=rot_gamma,
        )

    else:
        if type(rot_update) == np.ndarray:
            rot_update = torch.from_numpy(rot_update.astype(np.float32))

        rot_mat = axis_angle_to_matrix(rot_update.squeeze())

    if type(tr_update) == np.ndarray:
        tr_update = torch.from_numpy(tr_update.astype(np.float32))

    rigid_new_pos = (
        (data["ligand"].pos - ligand_center) @ rot_mat.T + tr_update + ligand_center
    )

    ############################ TORSIONAL GUIDING #############################
    # --------------------------------------------------------------------------#

    if tor_updates is not None:

        if tor_guiding and not in_torus_region(tau=tor_state, region=desired_tor):
            tor_vdir, tor_distance = tor_guider(
                state=tor_state,
                region=desired_tor,
            )

            if neg_vdir: # necessary because of how DiffDock applies the update
                tor_vdir = -tor_vdir

            if (
                mask_n_tor and mask_n_tor > 0
            ): 
                # mask part of the angles and doesn't apply guidance
                # this can be useful for regions with lower confidence or in the 
                # case that there are several regions possible for one angle and they
                # are conflicting
                n_angles = len(tor_state)
                if mask_n_tor > n_angles:
                    mask_n_tor = n_angles
                if mask_n_distribution == "random":
                    unique_indices = random.sample(range(n_angles), mask_n_tor)
                elif mask_n_distribution == "weighted":
                    p = desired_tor[-1][0]
                    p_norm = p / np.sum(p)
                    unique_indices = np.random.choice(
                        range(n_angles), size=mask_n_tor, replace=False, p=p_norm
                    )

                for index in unique_indices:
                    tor_vdir[index] = 0

            if dynamic_gamma:
                tor_gamma, tor_sim = compute_tor_gamma(
                    tor_updates, tor_vdir, tor_distance
                )

            tor_updates = get_guided_tor_update(
                current_state=tor_state,
                vdir=tor_vdir,
                distance=tor_distance,
                update=tor_updates,
                gamma=tor_gamma,
                update_method=Rm_update_method,
            )

        flexible_new_pos = modify_conformer_torsion_angles(
            rigid_new_pos,
            data["ligand", "ligand"].edge_index.T[data["ligand"].edge_mask],
            (
                data["ligand"].mask_rotate
                if isinstance(data["ligand"].mask_rotate, np.ndarray)
                else data["ligand"].mask_rotate[0]
            ),
            tor_updates,
        ).to(rigid_new_pos.device)

        R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)

        aligned_flexible_pos = flexible_new_pos @ R.T + t.T
        data["ligand"].pos = aligned_flexible_pos

    else:
        data["ligand"].pos = rigid_new_pos

    return data


def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py"""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embedding_size // 2) * scale, requires_grad=False
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == "sinusoidal":
        emb_func = lambda x: sinusoidal_embedding(embedding_scale * x, embedding_dim)
    elif embedding_type == "fourier":
        emb_func = GaussianFourierProjection(
            embedding_size=embedding_dim, scale=embedding_scale
        )
    else:
        raise NotImplemented
    return emb_func


def get_t_schedule(inference_steps):
    return np.linspace(1, 0, inference_steps + 1)[:-1]


def set_time(complex_graphs, t_tr, t_rot, t_tor, batchsize, all_atoms, device):
    complex_graphs["ligand"].node_t = {
        "tr": t_tr * torch.ones(complex_graphs["ligand"].num_nodes).to(device),
        "rot": t_rot * torch.ones(complex_graphs["ligand"].num_nodes).to(device),
        "tor": t_tor * torch.ones(complex_graphs["ligand"].num_nodes).to(device),
    }
    complex_graphs["receptor"].node_t = {
        "tr": t_tr * torch.ones(complex_graphs["receptor"].num_nodes).to(device),
        "rot": t_rot * torch.ones(complex_graphs["receptor"].num_nodes).to(device),
        "tor": t_tor * torch.ones(complex_graphs["receptor"].num_nodes).to(device),
    }
    complex_graphs.complex_t = {
        "tr": t_tr * torch.ones(batchsize).to(device),
        "rot": t_rot * torch.ones(batchsize).to(device),
        "tor": t_tor * torch.ones(batchsize).to(device),
    }
    if all_atoms:
        complex_graphs["atom"].node_t = {
            "tr": t_tr * torch.ones(complex_graphs["atom"].num_nodes).to(device),
            "rot": t_rot * torch.ones(complex_graphs["atom"].num_nodes).to(device),
            "tor": t_tor * torch.ones(complex_graphs["atom"].num_nodes).to(device),
        }
