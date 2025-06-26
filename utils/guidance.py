import torch
import math

import numpy as np

from numpy.typing import ArrayLike

from utils.geometry import axis_angle_to_matrix
from utils.torus_geodesics import in_circ_sector, min_tau, vdir_min_path
from utils.torsion import TorsionCalculator


class Scheduler:
    """
    Scheduler for the gamma parameter
    """
    def __init__(self, **kwargs):
        self.params = kwargs

    def constant(self, N: int) -> list:
        a = self.params.get("a", 0.2)
        return [a for _ in range(N)]

    def linear(self, N: int) -> list:
        return [i / N for i in range(N, 0, -1)]

    def sqrt_scaled_linear(self, N: int) -> list:
        return [i / N * (np.sqrt(N) / 10) for i in range(N, 0, -1)]

    def logarithmic(self, N: int) -> list:
        return [1 - math.log(1 + i) / math.log(1 + N) for i in range(N)]

    def quadratic(self, N: int) -> list:
        return [(1 - i / N) ** 2 for i in range(N)]

    def exponential(self, N: int) -> list:
        a = self.params.get("a", 0.1)
        return [1 - a ** (i / N) for i in range(N, 0, -1)]

    def sigmoid(self, N: int) -> list:
        k = self.params.get("k", 0.1)
        return [1 / (1 + math.exp(-k * (i - N / 2))) for i in range(N)]

    def inverted_sinusoidal(self, N: int) -> list:
        return [1 - (1 + math.sin(math.pi * i / N)) / 2 for i in range(N)]

    def piecewise_linear(self, N: int) -> list:
        midpoint = N // 2
        first_half = [2 * i / N for i in range(midpoint, 0, -1)]
        second_half = [i / N for i in range(midpoint, 0, -1)]
        return first_half + second_half

    def cosine_annealing(self, N: int) -> list:
        return [(1 + math.cos(math.pi * i / N)) / 2 for i in range(N)]

    def step(self, N: int) -> list:
        num_levels = self.params.get("num_levels", N // 5)
        steps_per_level = N // num_levels
        levels = [1 - i / num_levels for i in range(num_levels)]
        schedule = []
        for level in levels:
            schedule.extend([level] * steps_per_level)
        return schedule

    def warmup_cooldown(self, N: int) -> list:
        a = self.params.get("a", 0.3)
        n_begin = self.params.get("n_begin", N // 10)
        n_end = self.params.get("n_end", N // 10)
        base = [a for _ in range(N)]
        base[:n_begin] = [0] * n_begin
        base[-n_end:] = [0] * n_end
        return base

    def onoff(self, N: int) -> list:
        a = self.params.get("a", 0.3)
        last_n = self.params.get("last_n", 5)
        base = [a for _ in range(N)]
        onoff = [
            0 if i % 2 == 1 or i >= len(base) - last_n else base[i]
            for i in range(len(base))
        ]
        return onoff


class GammaScheduler:
    @staticmethod
    def get_schedule(scheduler, N, **kwargs):
        scheduler_obj = Scheduler(**kwargs)
        method = getattr(scheduler_obj, scheduler, None)
        if method is None:
            raise ValueError(f"Scheduler '{scheduler}' not found.")
        # Directly calling the method with N and returning the schedule
        return method(N)


# dynamic gammas
def _scale_dir(sim, k=5, g_cap=1):
    """scaled gamma based on cosine similarity of direction vectors"""
    return g_cap * 1 / (1 + np.exp(k * sim))


def _scale_dist(dist, max_dist, k=5, g_cap=1):
    """scale gamma based on distance"""
    normalized_distance = dist / max_dist
    return g_cap * (1 - np.exp(-k * normalized_distance))


def scale_gamma(sim, dist, max_dist, a=0.5):
    """scale gamma with a combination of distance and direction"""
    direction = _scale_dir(sim)
    dist = _scale_dist(dist, max_dist)

    return (1 - a) * direction + a * dist


def compute_tr_gamma(
    protein_diameter,
    tr_update,
    vdir,
    distance,
):
    """Computes tr gamma dynamically

    protein_diameter is used as max distance, assuming that the maximum distance possible
    happens when the ligand is in the polar opposite of its binding site

    This assumption is not completely true in the very early steps of diffusion, as
    same samples may fall outside the protein itself, but quickly falls inside of it.
    """
    tr_update = np.array(list(tr_update[0]))
    max_d = protein_diameter

    update_dist = np.linalg.norm(tr_update)
    update_dir = tr_update / update_dist

    sim = np.dot(update_dir, vdir[0])  # already normalized, sim is just the dot product

    gamma = scale_gamma(sim, distance, max_d)
    gamma = float(gamma)

    return gamma, sim


def compute_tor_gamma(
    tor_update,
    vdir,
    distance,
):
    """Computes tor_gamma dynamically.

    The distance between points in a m-dimensional torus is given by Pythagoras
    theorem in m dimensions:

    d = \sqrt{d_0^2 * d_1^2 * ... * d_m^2},

    Where d_0 is the contribution of each dimension (circle) of the torus

    To get the maximum distance, every distance d_i must be the maximum distance
    possible in a circle (pi), therefore:

    d_max = \sqrt{m * pi^2},
    d_max = pi * \sqrt{m}
    """

    update_dist = np.linalg.norm(tor_update)
    update_dir = tor_update / update_dist

    sim = np.dot(update_dir, vdir)
    max_d = math.pi * math.sqrt(len(vdir))
    gamma = scale_gamma(sim, distance, max_d)

    return gamma, sim


### TRANSLATIONAL GUIDANCE ###
def get_tr_state(positions):
    if isinstance(positions, torch.Tensor):
        positions = positions.numpy()

    return np.mean(positions, axis=0)


def in_tr_region(state, region):
    return np.linalg.norm(state.cpu().numpy() - region[0:3]) < region[3]


def tr_guider(pos, sph):
    # gets which direction to guide diffusion in towards the specified sphere, gets distance to the sphere's boundary
    vdir = sph[0:3] - pos
    aux = np.linalg.norm(vdir)
    vdir = vdir / aux

    # solve 2nd degree equation for lambda coefficient, which determines which point of the segment pos-sphere[0:3]
    # intersects the sphere's boundary
    a = (
        np.linalg.norm(pos) ** 2
        + np.linalg.norm(sph[0:3]) ** 2
        - 2 * np.dot(pos, sph[0:3])
    )
    b = -2 * a
    c = a - sph[3] ** 2
    l1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    l2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    if 0 <= l1 <= 1:
        dist = aux * l1
    else:
        dist = aux * l2

    alfa = aux  # aux aims for the sphere's center, dist aims for the sphere's boundary.
    return vdir, aux


def get_guided_tr_update(*args, **kwargs):
    return _get_guided_Rm_update(*args, **kwargs)


### ROTATIONAL GUIDANCE ###
def get_rot_state(positions, i=0, j=1):
    """Gets rotational state of a molecule given its conformer positions

    The rotational state is defined with two vectors:
     - v1: vector from the ligand center to the first atom
     - v3: vector perpendicular to v1 and v2, where v2 is
            the vector from the ligand center to the second atom"""
    if isinstance(positions, torch.Tensor):
        positions = positions.numpy()

    center = np.mean(positions, axis=0)
    v1 = positions[i] - center
    v2 = positions[j] - center
    v3 = np.cross(v1, v2)

    if np.linalg.norm(v3) == 0:
        v2 = (
            positions[j] + center * 0.1 - center
        )  # displace 2nd atom slightly to avoid both vectors being parallel
        v3 = np.cross(v1, v2)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    return np.array([v1, v3])


def in_rot_region(state: np.array, region):
    """Checks if the current state is inside the desired region, defined as
    boundaries for the theta and phi angles in the polar coordinate system"""
    dim = len(state)
    in_region = False
    for i in range(dim):  # 0, 1
        theta = np.arccos(state[i, 2])
        phi = math.atan2(state[i, 1] / np.sin(theta), state[i, 0] / np.sin(theta))
        if phi < 0:
            phi = 2 * math.pi + phi

        if region[i * 2 + 1, 0] <= region[i * 2 + 1, 1]:  # phi_m < phi_M
            if (
                region[i * 2, 0] <= theta <= region[i * 2, 1]
                and region[i * 2 + 1, 0] <= phi <= region[i + 1, 1]
            ):
                in_region = True
                break
        else:
            if region[i * 2, 0] <= theta <= region[i * 2, 1] and (
                region[i * 2 + 1, 0] <= phi or phi <= region[i + 1, 1]
            ):
                in_region = True
                break

    return in_region


def rot_guider(current_state, desired_rot):
    """Vector pointing at the closest point in the region* to the current state

    *currently computes the center of the region, not the closest point of the boundary
    """

    vdir = [[0, 0, 0], [0, 0, 0]]  # list to hold both vdirs
    distance = [10, 10]
    dim = len(current_state)  # 2: 0,1
    for i in range(dim):
        theta = (desired_rot[i * 2, 0] + desired_rot[i * 2, 1]) / 2
        phi = (desired_rot[i * 2 + 1, 0] + desired_rot[i * 2 + 1, 1]) / 2
        aux = [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
        if np.arccos(np.dot(current_state[i], vdir[i])) < distance[i]:
            vdir[i] = aux
            distance[i] = np.arccos(np.dot(current_state[i], vdir[i]))

    return vdir, distance


def get_guided_rotation_matrix(
    current_state,
    vdir,
    distance,
    rot_update,
    gamma: float,
):

    rot_update = rot_update.cpu().numpy()
    rot_update = (1 - float(gamma)) * rot_update
    rot_update = torch.from_numpy(rot_update.astype(np.float32))

    guided_updates = []
    dim = len(current_state)
    for i in range(dim):
        u = np.asarray(vdir[i]) - np.dot(current_state[i], vdir[i]) * current_state[i]
        u = u / np.linalg.norm(u)
        # (current_state[i], u) are the orthogonal basis of the plane that contains
        # the geodesic connecting current_state[i] and vdir[i]
        w = np.cross(
            current_state[i], u
        )  # rotation axis to align current_state[i] and vdir[i]
        guided_rot_update = (
            gamma * distance[i] * w
        )  # rotation to align current_state[i] and vdir[i] scaled by gamma
        guided_rot_update = torch.from_numpy(guided_rot_update.astype(np.float32))
        guided_updates.append(guided_rot_update)

    rot_mat = torch.matmul(
        torch.matmul(
            axis_angle_to_matrix(rot_update.squeeze()),
            axis_angle_to_matrix(guided_updates[0]),
        ),
        axis_angle_to_matrix(guided_updates[1]),
    )  # sequential application of rotation matrices

    return rot_mat


### TORSIONAL GUIDANCE ###


def get_tor_state(
    positions,
    edge_index,
    edge_mask,
    mol,
    method,
):

    tau = TorsionCalculator.calc_torsion_angles(
        positions=positions,
        edge_index=edge_index,
        edge_mask=edge_mask,
        mol=mol,
        method=method,
    )

    return tau


def in_torus_region(tau: ArrayLike, region: ArrayLike) -> bool:
    """Checks if a torsional state is inside given regions in a hypertorus

    Parameters
    ----------
    tau : ArrayLike
        torsional state.
        Array of m values that represent coordinates in a m dimensional hypertorus
    region : ArrayLike
        regions in the hypertorus.
        np.array with shape m x 2n, where m is the number of dimensions in the hypertorus
        (and therefore the number of rotatable bonds) and n is the number of regions for
        each angle
    """
    for i in range(len(tau)):
        in_any_sector = False
        for j in range(len(region[i]) // 2):
            if in_circ_sector(tau[i], region[i][2 * j : 2 * j + 2]):
                in_any_sector = True
                break
        if not in_any_sector:
            return False
    return True


def tor_guider(state, region):
    """
    gets torsional diffusion guiding vector, which defines the minimum path between
    the current state and the border of the desired region (boundary vector)
    """
    dim = len(state)
    tau, closest_region = min_tau(state, region)
    vdir, dist = vdir_min_path(state, tau)
    return vdir, dist


def get_guided_tor_update(*args, **kwargs):
    return _get_guided_Rm_update(*args, **kwargs)


def _get_guided_Rm_update(
    current_state,
    vdir,
    distance,
    update,
    gamma,
    update_method="m0",
):
    """Gets guided updates in R^m. This works for both translation and torsion,
    since torsion is just R^m mod 2pi
    """

    update_dist = np.linalg.norm(update)
    update_dir = update / update_dist

    method_dict = {
        "m0": UpdateCalculator.m0,
        "m1": UpdateCalculator.m1,
        "m2": UpdateCalculator.m2,
        "m3": UpdateCalculator.m3,
        "m4": UpdateCalculator.m4,
    }

    guided_updates = method_dict[update_method](
        update_dir=update_dir,
        update_dist=update_dist,
        vdir=vdir,
        distance=distance,
        gamma=gamma,
    )

    return guided_updates


class UpdateCalculator:
    """used for benchmarking, in practice only m0 needed"""

    @staticmethod
    def m0(update_dir, update_dist, vdir, distance, gamma):
        """original"""
        guided_update = (1 - gamma) * update_dir * update_dist + gamma * vdir * distance
        return guided_update

    @staticmethod
    def m1(update_dir, update_dist, vdir, distance, gamma):
        """original w/o distance"""
        guided_update = (1 - gamma) * update_dir * update_dist + gamma * vdir
        return guided_update

    @staticmethod
    def m2(update_dir, update_dist, vdir, distance, gamma):
        """guiding direction * distance"""

        guided_dir = (1 - gamma) * update_dir + gamma * vdir
        guided_update = guided_dir * distance

        return guided_update

    @staticmethod
    def m3(update_dir, update_dist, vdir, distance, gamma):
        """guiding direction * update magnitude"""
        guided_dir = (1 - gamma) * update_dir + gamma * vdir
        guided_update = guided_dir * update_dist

        return guided_update

    @staticmethod
    def m4(update_dir, update_dist, vdir, distance, gamma):
        """guiding independently"""

        guided_dir = (1 - gamma) * update_dir + gamma * vdir
        guided_dist = (1 - gamma) * update_dist + gamma * distance

        guided_update = guided_dir * guided_dist

        return guided_update
