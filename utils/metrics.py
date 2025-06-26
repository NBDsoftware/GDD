"""Metrics to evaluate docking poses"""

import os
import math
import json
import ipdb

import numpy as np
import pandas as pd

from typing import List

from utils.torsion import TorsionCalculator, AngleCalcMethod
from utils.torus_geodesics import min_path
from utils.utils import get_symmetry_rmsd
from utils.guidance import get_rot_state


class Pose:
    def __init__(
        self,
        mol=None,
        pos=None,
        edge_index=None,
        edge_mask=None,
        tr_state=None,
        rot_state=None,
        tor_state=None,
    ):
        self.mol = mol
        self.pos = pos
        self.edge_index = edge_index
        self.edge_mask = edge_mask
        self.tr_state = tr_state if tr_state is not None else self._get_tr_state()
        self.rot_state = rot_state if rot_state is not None else self._get_rot_state()
        self.tor_state = tor_state if tor_state is not None else self._get_tor_state()

    def _get_tr_state(self):
        return np.mean(self.pos, axis=0)

    def _get_rot_state(self):
        return get_rot_state(self.pos, i=0, j=1)

    def _get_tor_state(self):
        tau = TorsionCalculator.calc_torsion_angles(
            positions=self.pos,
            edge_index=self.edge_index,
            edge_mask=self.edge_mask,
        )
        return tau

    def save_states():
        raise NotImplementedError


class MetricsCalculator:

    @staticmethod
    def save_metrics(metrics: dict, path: str):
        """Save metrics in a json file"""

        # create dir if it doesnt exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f)

    @staticmethod
    def load_metrics(path: str):
        with open(path, "r") as f:
            metrics = json.load(f)
        return metrics

    @staticmethod
    def compute_all_metrics(s1: Pose, s2: List[Pose]) -> dict:
        metrics = {
            "rmsd": [],
            "tr_metrics": {"tr_distance": []},
            "rot_metrics": {
                "rot_angles_mae": [],
                "rot_angles_mse": [],
                "rot_angles_rmse": [],
                "rot_state_mean_cosim": [],
            },
            "tor_metrics": {
                "tor_angles_mae": [],
                "tor_angles_mse": [],
                "tor_angles_rmse": [],
                "tor_dist": [],
            },
        }

        if not isinstance(s2, list):
            s2 = [s2]

        for state in s2:
            metrics["rmsd"].append(MetricsCalculator._compute_rmsd(s1, state))

            tr_metric = MetricsCalculator._compute_tr_metrics(s1, state)
            for key in tr_metric:
                metrics["tr_metrics"][key].append(tr_metric["tr_distance"])

            rot_metric = MetricsCalculator._compute_rot_metrics(s1, state)
            for key in rot_metric:
                metrics["rot_metrics"][key].append(rot_metric[key])

            tor_metric = MetricsCalculator._compute_tor_metrics(s1, state)
            for key in tor_metric:
                metrics["tor_metrics"][key].append(tor_metric[key])

        return metrics

    @staticmethod
    def _compute_rmsd(s1, s2):
        rmsd = get_symmetry_rmsd(s1.mol, s1.pos, s2.pos)
        return rmsd

    @staticmethod
    def _compute_tr_metrics(s1, s2):
        metrics = {}
        metrics["tr_distance"] = tr_states_distance(s1, s2)
        return metrics

    @staticmethod
    def _compute_rot_metrics(s1, s2):
        metrics = {}
        d_rot = rot_angles_differences(s1, s2)
        metrics["rot_angles_mae"] = mae(d_rot)
        metrics["rot_angles_mse"] = mse(d_rot)
        metrics["rot_angles_rmse"] = rmse(d_rot)
        metrics["rot_state_mean_cosim"] = mean_cosine_similarity(
            s1.rot_state, s2.rot_state
        )
        return metrics

    @staticmethod
    def _compute_tor_metrics(s1, s2):
        metrics = {}
        d_tor = tor_angles_differences(s1, s2)
        if len(d_tor) == 0:  # no torsion angles
            return metrics
        metrics["tor_angles_mae"] = mae(d_tor)
        metrics["tor_angles_mse"] = mse(d_tor)
        metrics["tor_angles_rmse"] = rmse(d_tor)
        metrics["tor_dist"] = distance_in_torus(s1, s2)
        return metrics


class DiffDockResultSet:
    def __init__(
        self,
        path,
        get_tables=False,
        name=None,
        tr_gamma=None,
        rot_gamma=None,
        tor_gamma=None,
        steps=None,
        add_hs_conformer=None,
        remove_hs_mol=None,
        reordering=None,
        angle_calc_method=None,
        update_method=None,
        scheduler=None,
        margin=None,
        mask=None,
        **kwargs,
    ):
        self.path = path
        self.name = name if name else path
        
        self.tr_gamma = tr_gamma
        self.rot_gamma = rot_gamma
        self.tor_gamma = tor_gamma
        self.add_hs_conformer = add_hs_conformer
        self.remove_hs_mol = remove_hs_mol
        self.reordering = reordering
        self.angle_calc_method = angle_calc_method
        self.update_method = update_method
        self.scheduler = scheduler
        self.margin = margin
        self.mask = mask
        self.steps = steps

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.metrics_dict = self._get_metrics_dict()
        self.aggregated_metrics_dict = self._get_aggregated_metrics_dict()
        if get_tables:
            self.metrics_dataframe = self._get_metrics_dataframe()

    def _get_metrics_dict(self):
        metrics_dict = {}
        compounds_dirs = os.listdir(
            self.path
        )  # either complex names (pdb ids) as pdbid.json or folders
        if not self.inter_steps:
            metrics_dict[f"step{self.steps-1}"] = {}
            if compounds_dirs[0].endswith(".json"):
                for file in compounds_dirs:  
                    compound = file.split(".")[0]
                    filename = os.path.join(self.path, file)
                    metrics_dict[f"step{self.steps-1}"][compound] = MetricsCalculator.load_metrics(filename)
            else:
                # complex names as folders, with intermediary steps saved
                for compound in compounds_dirs:
                    metrics_dict[compound] = {}
                    for file in os.listdir(os.path.join(self.path, compound)):  # stepX.json
                        step = file[:-5]
                        filename = os.path.join(self.path, compound, step + ".json")
                        metrics_dict[compound][step] = MetricsCalculator.load_metrics(filename)
        else:
            for step in range(self.steps):
                step_name = f"step{step}"
                metrics_dict[step_name] = {}
                for compound in compounds_dirs:
                    try:
                        filename = os.path.join(self.path, compound, step_name + ".json")
                        metrics_dict[step_name][compound] = (MetricsCalculator.load_metrics(filename))
                    except:
                        continue

        metrics_dict = self._flatten_metrics_dict(metrics_dict)
        return metrics_dict

    def _flatten_metrics_dict(self, metrics_dict):
        flat_metrics = {}
        for step, results_at_step in metrics_dict.items():
            flat_metrics[step] = {}
            for compound, results_for_compound in results_at_step.items():
                flat_metrics[step][compound] = {}
                for metric_name, metric_values in results_for_compound.items():
                    if metric_name in ["tr_metrics", "rot_metrics", "tor_metrics"]:
                        for submetric_name, submetric_values in metric_values.items():
                            flat_metrics[step][compound][submetric_name] = submetric_values
                    else:
                        flat_metrics[step][compound][metric_name] = metric_values

        return flat_metrics

    def _get_aggregated_metrics_dict(self):

        metric_keys = ["top_hits", "top_fives", "means", "medians", "mins", "all"]
        steps = [f"step{i}" for i in range(self.steps)]

        # 1. Initialize structure using the first item
        first_metrics_step = self.metrics_dict[next(iter(self.metrics_dict))]
        first_metrics = first_metrics_step[next(iter(first_metrics_step))]
        aggregated_metrics_dict = {
            step: {metric: {} for metric in first_metrics.keys()} for step in steps
        }

        # structure now is: {'rmsd': {}, 'tr_metrics': {}, 'rot_metrics': {}, 'tor_metrics': {}}
        # with steps: Structure now is: {'step0': {'rmsd': {}, 'tr_metrics': {}, 'rot_metrics': {}, 'tor_metrics': {}}, 'step1': {...}, ...}

        for step, metrics_dict in self.metrics_dict.items():
            for metric_key, metric_value in metrics_dict.items():
                for metric in metric_value:
                    if isinstance(metric_value[metric], dict):
                        for sub_metric in metric_value[metric]:
                            aggregated_metrics_dict[step][metric][sub_metric] = {
                                key: [] for key in metric_keys
                            }
                    else:
                        aggregated_metrics_dict[step][metric] = {
                            key: [] for key in metric_keys
                        }

        # structure now is:
        # {
        #   'rmsd': values,
        #   'tr_metrics': {
        #       'tr_metric_0': values,
        #   }
        #   'rot_metrics': {
        #       'rot_metric_0': values,
        #       ...
        #   }
        #   'tor_metrics' : {
        #       ...
        #   }

        # With steps, structure now is:
        # {
        #   'step0': {
        #       'rmsd': values,
        #       'tr_metrics': {
        #           'tr_metric_0': values,
        #       }
        #       'rot_metrics': {
        #           'rot_metric_0': values,
        #           ...
        #       }
        #       'tor_metrics' : {
        #           ...
        #       }
        #   },
        #   'step1': {...},
        #   ...
        # }

        # where values = {'top_hits': [], 'top_fives': [], 'means': [], 'medians': [], 'mins': [], 'all': []}

        # 2. Aggregate metrics
        for step, compounds_metrics in self.metrics_dict.items():
            for compound, metrics in compounds_metrics.items():
                for metric_name, metric_values in metrics.items():
                    if isinstance(metric_values, dict):
                        for sub_metric, values in metric_values.items():
                            try:
                                aggregated_metrics_dict[step][metric_name][sub_metric]["top_hits"].append(values[0])
                            except:  # found compound with no torsions
                                continue
                            aggregated_metrics_dict[step][metric_name][sub_metric]["top_fives"].append(np.min(values[:5]))
                            aggregated_metrics_dict[step][metric_name][sub_metric]["mins"].append(np.min(values))
                            aggregated_metrics_dict[step][metric_name][sub_metric]["means"].append(np.mean(values))
                            aggregated_metrics_dict[step][metric_name][sub_metric]["medians"].append(np.median(values))
                            aggregated_metrics_dict[step][metric_name][sub_metric]["all"].extend(values)
                    else:
                        try:
                            aggregated_metrics_dict[step][metric_name]["top_hits"].append(metric_values[0])
                        except:  # compounds that failed for some reason (in diffdock)
                            continue

                        aggregated_metrics_dict[step][metric_name]["top_fives"].append(np.min(metric_values[:5]))
                        aggregated_metrics_dict[step][metric_name]["mins"].append(np.min(metric_values))
                        aggregated_metrics_dict[step][metric_name]["means"].append(np.mean(metric_values))
                        aggregated_metrics_dict[step][metric_name]["medians"].append(np.median(metric_values))
                        aggregated_metrics_dict[step][metric_name]["all"].extend(metric_values)

        return aggregated_metrics_dict

    def _set_average_metrics(self):
        for key, values in self.metrics_dict.items():
            average_value = np.mean(values)
            setattr(self, f"avg_{key}", average_value)

    def _get_rmsd_dataframe(self):

        def percentage_below(values, threshold):
            return np.sum(np.array(values) < threshold) / len(values) * 100

        df = pd.DataFrame(
            columns=["Step", "Metric", "Average", "Median", "Below 2%", "Below 5%"]
        )

        rows = []

        for step, metrics_dict in self.aggregated_metrics_dict.items():
            for metric, values in metrics_dict.get("rmsd", {}).items():
                avg = np.mean(values)
                median = np.median(values)
                below_2 = percentage_below(values, 2) if metric == "rmsd" else None
                below_5 = percentage_below(values, 5) if metric == "rmsd" else None

                row = {
                    "Step": step,
                    "Metric": metric,
                    "Average": avg,
                    "Median": median,
                    "% Below 2A": below_2,
                    "% Below 5A": below_5,
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def _get_metrics_dataframe(self):

        def percentage_below(values, threshold):
            return np.sum(np.array(values) < threshold) / len(values) * 100

        df = pd.DataFrame(
            columns=["Step", "Metric", "Average", "Median", "Below 2%", "Below 5%"]
        )

        rows = []

        for step, metrics_dict in self.aggregated_metrics_dict.items():
            for metric, aggregated_metrics_dict in metrics_dict.items():
                for aggregated_metric, values in aggregated_metrics_dict.items():
                    avg = np.nanmean(values)
                    median = np.nanmedian(values)
                    below_2 = percentage_below(values, 2) if metric == "rmsd" else None
                    below_5 = percentage_below(values, 5) if metric == "rmsd" else None

                    row = {
                        "Step": step,
                        "Metric": metric,
                        "Aggregated Metric": aggregated_metric,
                        "Average": avg,
                        "Median": median,
                        "% Below 2A": below_2,
                        "% Below 5A": below_5,
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def set_parameters(
        self,
        tr_gamma=None,
        rot_gamma=None,
        tor_gamma=None,
        add_hs_conformer=None,
        remove_hs_mol=None,
        reorder=None,
        tor_calc_method=None,
    ):
        self.tr_gamma = tr_gamma
        self.rot_gamma = rot_gamma
        self.tor_gamma = tor_gamma
        self.add_hs_conformer = add_hs_conformer
        self.remove_hs_mol = remove_hs_mol
        self.reorder = reorder
        self.tor_calc_method = tor_calc_method

    def __str__(self):
        return f"ResultSet for {self.path}"

    def __repr__(self):
        return str(self)


## metrics functions ##

# misc.


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def mean_cosine_similarity(a1, a2):
    """pairwise mean cosine similarity between two arrays of vectors"""
    assert len(a1) == len(a2)
    return np.mean([cosine_similarity(a1[i], a2[i]) for i in range(len(a1))])


def mae(d):
    return np.mean(np.abs(d))


def mse(d):
    return np.mean(np.square(d))


def rmse(d):
    return np.sqrt(np.mean(np.square(d)))


def max_absolute_error(differences):
    return np.max(np.abs(differences))


# translation


def tr_states_distance(s1: Pose, s2: Pose):
    """distance between the centers"""
    return np.linalg.norm(s1.tr_state - s2.tr_state)


# rotation


def get_rot_angles(v):
    theta = np.arccos(v[2])
    phi = math.atan2(v[1] / np.sin(theta), v[0] / np.sin(theta))
    if phi < 0:
        phi = 2 * math.pi + phi
    # phi = math.atan2(v[1], v[0])
    return theta, phi


def rot_angles_differences(s1: Pose, s2: Pose):
    d = []
    for s1_vec, s2_vec in zip(s1.rot_state, s2.rot_state):
        theta1, phi1 = get_rot_angles(s1_vec)
        theta2, phi2 = get_rot_angles(s2_vec)
        d.extend([theta1 - theta2, phi1 - phi2])

    return d


def rot_states_mean_cosine_similarity(s1: Pose, s2: Pose):
    """Rot state has two vectors v1 and v2.
    This function computes the mean cosine similarity beteen (s1_v1, s2_v1)
    and (s1_v2, s2_v2)
    """
    sim = mean_cosine_similarity(s1.rot_state, s2.rot_state)
    return sim


# torsion

def tor_angles_differences(s1: Pose, s2: Pose):
    if len(s1.tor_state) != len(s2.tor_state):
        raise ValueError("Torsion lists must be of the same length.")

    if len(s1.tor_state) == 0:
        return []  # no torsion angles

    differences = []
    for angle1, angle2 in zip(s1.tor_state, s2.tor_state):
        diff = abs(angle1 - angle2)
        diff = min(diff, 2 * np.pi - diff)  # Account for periodicity in radians
        differences.append(diff)

    return differences


def distance_in_torus(s1: Pose, s2: Pose):
    return min_path(s1.tor_state, s2.tor_state)


def total_variational_distance(differences):
    return np.sum(differences)
