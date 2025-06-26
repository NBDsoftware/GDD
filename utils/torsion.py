import torch
import copy
import math

import networkx as nx
import sknetwork as skn
import numpy as np

from enum import Enum
from typing import List, Tuple

from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx, to_scipy_sparse_matrix
from torch_geometric.data import Data

"""
    Preprocessing and computation for torsional updates to conformers
"""


class AngleCalcMethod(Enum):
    TOR_CALC_1 = "tor_calc_1"
    TOR_CALC_2 = "tor_calc_2"


class TorsionCalculator:

    @staticmethod
    def _with_all_neighbors(positions, edge_index, edge_mask):
        """Computes angles using all neighbors

        This means that for the bond defined by:
        (a1, a2, a3 ) - b - c - (d1, d2, d3)

        The resulting angle is computed without choosing a or d. This has the
        advantage of not having to do so, but the disadvantage that different 
        positions of (a1, a2, a3) can give the same torsion value
        """
        ADJ = to_scipy_sparse_matrix(edge_index)
        ADJ = ADJ.tocsr()

        Tau = []
        if type(positions) != np.ndarray:
            positions = positions.cpu().numpy()
        for edge_idx, edge in enumerate(edge_index.T[edge_mask]):
            b, c = edge[0], edge[1]
            nb, nc = skn.utils.get_neighbors(ADJ, b), skn.utils.get_neighbors(ADJ, c)
            a, d = [-1], [-1]
            COS = 0
            SIN = 0
            for i in nb:
                if i != c:
                    a = i
                    for j in nc:
                        if j != b:
                            d = j
                            ab = positions[b] - positions[a]
                            ab = ab / np.linalg.norm(ab)
                            bc = positions[c] - positions[b]
                            bc = bc / np.linalg.norm(bc)
                            cd = positions[d] - positions[c]
                            cd = cd / np.linalg.norm(cd)
                            abc = np.cross(ab, bc)
                            abc = abc / np.linalg.norm(abc)
                            bcd = np.cross(bc, cd)
                            bcd = bcd / np.linalg.norm(bcd)
                            SIN += np.dot(bc, np.cross(abc, bcd))
                            COS += np.dot(abc, bcd)

            s = [COS, SIN]
            if s == [0, 0]:
                Tau.append(math.e)
            else:
                s = s / np.linalg.norm(s)
                tor = math.atan2(s[1], s[0])
                if tor < 0:
                    tor = 2 * math.pi + tor

                Tau.append(tor)

        positions = torch.from_numpy(positions.astype(np.float32))

        return np.array(Tau)

    @staticmethod
    def _with_torsions(positions, torsions) -> np.ndarray:
        """Computes torsion angles given the bonds that define them
        (i.e. a-b-c-d atoms)
        """
        Tau = []
        if type(positions) != np.ndarray:
            positions = positions.cpu().numpy()
        for tor_idx, tors in enumerate(torsions):
            a, b, c, d = tors
            COS = 0
            SIN = 0
            ab = positions[b] - positions[a]
            ab = ab / np.linalg.norm(ab)
            bc = positions[c] - positions[b]
            bc = bc / np.linalg.norm(bc)
            cd = positions[d] - positions[c]
            cd = cd / np.linalg.norm(cd)
            abc = np.cross(ab, bc)
            abc = abc / np.linalg.norm(abc)
            bcd = np.cross(bc, cd)
            bcd = bcd / np.linalg.norm(bcd)
            SIN += np.dot(bc, np.cross(abc, bcd))
            COS += np.dot(abc, bcd)

            s = [COS, SIN]
            if s == [0, 0]:
                Tau.append(math.e)
            else:
                s = s / np.linalg.norm(s)
                tor = math.atan2(s[1], s[0])
                if tor < 0:
                    tor = 2 * math.pi + tor

                Tau.append(tor)

        positions = torch.from_numpy(positions.astype(np.float32))
        return np.array(Tau)

    @staticmethod
    def calc_torsion_angles(
        positions,
        edge_index,
        edge_mask,
        mol=None,
        method: AngleCalcMethod = AngleCalcMethod.TOR_CALC_2,
    ):

        if method == AngleCalcMethod.TOR_CALC_1:
            return TorsionCalculator._with_all_neighbors(
                positions, edge_index, edge_mask
            )

        elif method == AngleCalcMethod.TOR_CALC_2:
            torsions = _get_torsions(edge_index, edge_mask)
            return TorsionCalculator._with_torsions(positions, torsions)


def _get_torsions(edge_index, edge_mask) -> List[Tuple[int, int, int, int]]:
    """
    from https://github.com/gcorso/torsional-diffusion

    Gets a list of the rotatable bonds expressed as tuples of atoms indexes.
    """
    edge_list = [[] for _ in range(torch.max(edge_index) + 1)]

    for p in edge_index.T:
        edge_list[p[0]].append(p[1])

    rot_bonds = [(p[0], p[1]) for i, p in enumerate(edge_index.T) if edge_mask[i]]

    dihedral = []
    for a, b in rot_bonds:
        c = edge_list[a][0] if edge_list[a][0] != b else edge_list[a][1]
        d = edge_list[b][0] if edge_list[b][0] != a else edge_list[b][1]
        dihedral.append((c.item(), a.item(), b.item(), d.item()))

    return dihedral


def get_transformation_mask(pyg_data):
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False)
    to_rotate = []
    edges = pyg_data["ligand", "ligand"].edge_index.T.numpy()
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


def modify_conformer_torsion_angles(
    pos, edge_index, mask_rotate, torsion_updates, as_numpy=False
):
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray:
        pos = pos.cpu().numpy()

    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = (
            rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec)
        )  # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (
            pos[mask_rotate[idx_edge]] - pos[v]
        ) @ rot_mat.T + pos[v]

    if not as_numpy:
        pos = torch.from_numpy(pos.astype(np.float32))
    return pos


def perturb_batch(data, torsion_updates, split=False, return_updates=False):
    if type(data) is Data:
        return modify_conformer_torsion_angles(
            data.pos,
            data.edge_index.T[data.edge_mask],
            data.mask_rotate,
            torsion_updates,
        )
    pos_new = [] if split else copy.deepcopy(data.pos)
    edges_of_interest = data.edge_index.T[data.edge_mask]
    idx_node = 0
    idx_edges = 0
    torsion_update_list = []
    for i, mask_rotate in enumerate(data.mask_rotate):
        pos = data.pos[idx_node : idx_node + mask_rotate.shape[1]]
        edges = (
            edges_of_interest[idx_edges : idx_edges + mask_rotate.shape[0]] - idx_node
        )
        torsion_update = torsion_updates[idx_edges : idx_edges + mask_rotate.shape[0]]
        torsion_update_list.append(torsion_update)
        pos_new_ = modify_conformer_torsion_angles(
            pos, edges, mask_rotate, torsion_update
        )
        if split:
            pos_new.append(pos_new_)
        else:
            pos_new[idx_node : idx_node + mask_rotate.shape[1]] = pos_new_

        idx_node += mask_rotate.shape[1]
        idx_edges += mask_rotate.shape[0]
    if return_updates:
        return pos_new, torsion_update_list
    return pos_new
