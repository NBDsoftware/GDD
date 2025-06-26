import math
import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike
from numba import jit


def in_circ_sector(alpha: float, sector: ArrayLike) -> bool:
    """
    checks wether an angle is within a circular sector
    takes into account periodicity (i.e. 0.5 is inside [2pi, 1])
    """
    if sector[1] >= sector[0]:
        return sector[0] <= alpha <= sector[1]
    else:
        return alpha >= sector[0] or alpha <= sector[1]


def min_path(z1, z2):
    """
    Computes the distance of shortest path between two points in a m-dimensional hypertorus
    This shortest path is equivalent to the shortest path for each of the m circles
    that form the hypertorus
    """
    l = 0

    direct_dist = np.abs(z2 - z1)
    wrap_dist = 2 * np.pi - direct_dist
    min_dist = np.where(direct_dist <= wrap_dist, direct_dist, wrap_dist)

    l = np.sqrt(np.sum(min_dist**2))

    return l


@jit(nopython=True) 
def vdir_min_path(tau1: np.ndarray, tau2: np.ndarray):
    """Computes v_dir, the tangent vector to the geodesic (shortest path) between
    two points in a m-dimensional hypertorus

    Returns both the vdir and the distance between the two points"""

    diff = tau2 - tau1
    complement = np.where(diff > 0, 2 * np.pi - diff, -2 * np.pi - diff)
    vdir = np.where(np.abs(diff) <= np.abs(complement), diff, complement)

    dist = np.linalg.norm(vdir)
    vdir_normalized = vdir / dist
    return vdir_normalized, dist


def min_tau(z: np.ndarray, region: ArrayLike) -> Tuple[np.ndarray, list]:
    """Computes the boundary vector of a set of regions given a point in a m dimensional hypertorus

    This boundary vector is a (m x 1) vector with the closest boundaries of each region to their
    corresponding angle

    (i.e. if z == [0, 1] and region==[[6, 3] [2, 4]], the boundary vector would be [6, 2])
    """
    tau = []
    closest_region = []
    for i in range(0, len(z)):
        min_length = 9999999  # valid for up to dim = 253300
        candidate = region[i, 0]
        rg_candidate = 0
        found = 0
        for j in range(0, len(region[i]) // 2):
            if found:
                continue
            if in_circ_sector(z[i], region[i, 2 * j : 2 * (j + 1)]):
                tau.append(z[i])
                closest_region.append(j)
                found = 1
            else:
                l1 = min_path(z[i], region[i, 2 * j])
                l2 = min_path(z[i], region[i, 2 * j + 1])
                if l1 <= l2:
                    if l1 < min_length:
                        min_length = l1
                        candidate = region[i, 2 * j]
                        rg_candidate = j
                else:
                    if l2 < min_length:
                        min_length = l2
                        candidate = region[i, 2 * j + 1]
                        rg_candidate = j

        if not found:
            tau.append(candidate)
            closest_region.append(rg_candidate)

    return np.array(tau), closest_region
