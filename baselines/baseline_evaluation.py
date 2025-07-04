# small script to extract the ligand and save it in a separate file because GNINA will use the ligand position as initial pose
import os

import plotly.express as px
import time
from argparse import FileType, ArgumentParser

import numpy as np
import pandas as pd
import wandb
from biopandas.pdb import PandasPdb
from rdkit import Chem

from tqdm import tqdm

from datasets.pdbbind import read_mol
from datasets.process_mols import read_molecule
from utils.utils import read_strings_from_txt, get_symmetry_rmsd

parser = ArgumentParser()
parser.add_argument("--config", type=FileType(mode="r"), default=None)
parser.add_argument("--run_name", type=str, default="gnina_results", help="")
parser.add_argument("--data_dir", type=str, default="data/PDBBind_processed", help="")
parser.add_argument(
    "--results_path",
    type=str,
    default="results/user_inference",
    help="Path to folder with trained model and hyperparameters",
)
parser.add_argument(
    "--file_suffix",
    type=str,
    default="_baseline_ligand.pdb",
    help="Path to folder with trained model and hyperparameters",
)
parser.add_argument("--project", type=str, default="ligbind_inf", help="")
parser.add_argument("--wandb", action="store_true", default=False, help="")
parser.add_argument("--file_to_exclude", type=str, default=None, help="")
parser.add_argument(
    "--all_dirs_in_results",
    action="store_true",
    default=True,
    help="Evaluate all directories in the results path instead of using directly looking for the names",
)
parser.add_argument("--num_predictions", type=int, default=10, help="")
parser.add_argument("--no_id_in_filename", action="store_true", default=False, help="")
args = parser.parse_args()

print("Reading paths and names.")
names = read_strings_from_txt(f"data/splits/timesplit_test")
names_no_rec_overlap = read_strings_from_txt(
    f"data/splits/timesplit_test_no_rec_overlap"
)
results_path_containments = os.listdir(args.results_path)

if args.wandb:
    wandb.init(
        entity="coarse-graining-mit",
        settings=wandb.Settings(start_method="fork"),
        project=args.project,
        name=args.run_name,
        config=args,
    )

all_times = []
successful_names_list = []
rmsds_list = []
centroid_distances_list = []
min_cross_distances_list = []
min_self_distances_list = []
without_rec_overlap_list = []
start_time = time.time()
for i, name in enumerate(tqdm(names)):
    mol = read_mol(args.data_dir, name, remove_hs=True)
    mol = Chem.RemoveAllHs(mol)
    orig_ligand_pos = np.array(mol.GetConformer().GetPositions())

    if args.all_dirs_in_results:
        directory_with_name = [
            directory for directory in results_path_containments if name in directory
        ][0]
        ligand_pos = []
        for i in range(args.num_predictions):
            file_paths = os.listdir(
                os.path.join(args.results_path, directory_with_name)
            )
            file_path = [path for path in file_paths if f"rank{i+1}" in path][0]
            if args.file_to_exclude is not None and args.file_to_exclude in file_path:
                continue
            mol_pred = read_molecule(
                os.path.join(args.results_path, directory_with_name, file_path),
                remove_hs=True,
                sanitize=True,
            )
            mol_pred = Chem.RemoveAllHs(mol_pred)
            ligand_pos.append(mol_pred.GetConformer().GetPositions())
        ligand_pos = np.asarray(ligand_pos)
    else:
        if not os.path.exists(
            os.path.join(
                args.results_path,
                name,
                f'{"" if args.no_id_in_filename else name}{args.file_suffix}',
            )
        ):
            raise Exception(
                "path did not exists:",
                os.path.join(
                    args.results_path,
                    name,
                    f'{"" if args.no_id_in_filename else name}{args.file_suffix}',
                ),
            )
        mol_pred = read_molecule(
            os.path.join(
                args.results_path,
                name,
                f'{"" if args.no_id_in_filename else name}{args.file_suffix}',
            ),
            remove_hs=True,
            sanitize=True,
        )
        if mol_pred == None:
            print("Skipping ", name, " because RDKIT could not read it.")
            continue
        mol_pred = Chem.RemoveAllHs(mol_pred)
        ligand_pos = np.asarray(
            [
                np.array(mol_pred.GetConformer(i).GetPositions())
                for i in range(args.num_predictions)
            ]
        )
    try:
        rmsd = get_symmetry_rmsd(
            mol, orig_ligand_pos, [l for l in ligand_pos], mol_pred
        )
    except Exception as e:
        print("Using non corrected RMSD because of the error:", e)
        rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))

    rmsds_list.append(rmsd)
    centroid_distances_list.append(
        np.linalg.norm(
            ligand_pos.mean(axis=1) - orig_ligand_pos[None, :].mean(axis=1), axis=1
        )
    )

    rec_path = os.path.join(args.data_dir, name, f"{name}_protein_processed.pdb")
    if not os.path.exists(rec_path):
        rec_path = os.path.join(
            args.data_dir, name, f"{name}_protein_obabel_reduce.pdb"
        )
    rec = PandasPdb().read_pdb(rec_path)
    rec_df = rec.df["ATOM"]
    receptor_pos = (
        rec_df[["x_coord", "y_coord", "z_coord"]]
        .to_numpy()
        .squeeze()
        .astype(np.float32)
    )
    receptor_pos = np.tile(receptor_pos, (args.num_predictions, 1, 1))

    cross_distances = np.linalg.norm(
        receptor_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1
    )
    self_distances = np.linalg.norm(
        ligand_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1
    )
    self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
    min_cross_distances_list.append(np.min(cross_distances, axis=(1, 2)))
    min_self_distances_list.append(np.min(self_distances, axis=(1, 2)))
    successful_names_list.append(name)
    without_rec_overlap_list.append(1 if name in names_no_rec_overlap else 0)
performance_metrics = {}
for overlap in ["", "no_overlap_"]:
    if "no_overlap_" == overlap:
        without_rec_overlap = np.array(without_rec_overlap_list, dtype=bool)
        rmsds = np.array(rmsds_list)[without_rec_overlap]
        centroid_distances = np.array(centroid_distances_list)[without_rec_overlap]
        min_cross_distances = np.array(min_cross_distances_list)[without_rec_overlap]
        min_self_distances = np.array(min_self_distances_list)[without_rec_overlap]
        successful_names = np.array(successful_names_list)[without_rec_overlap]
    else:
        rmsds = np.array(rmsds_list)
        centroid_distances = np.array(centroid_distances_list)
        min_cross_distances = np.array(min_cross_distances_list)
        min_self_distances = np.array(min_self_distances_list)
        successful_names = np.array(successful_names_list)

    np.save(os.path.join(args.results_path, f"{overlap}rmsds.npy"), rmsds)
    np.save(os.path.join(args.results_path, f"{overlap}names.npy"), successful_names)
    np.save(
        os.path.join(args.results_path, f"{overlap}min_cross_distances.npy"),
        np.array(min_cross_distances),
    )
    np.save(
        os.path.join(args.results_path, f"{overlap}min_self_distances.npy"),
        np.array(min_self_distances),
    )

    performance_metrics.update(
        {
            f"{overlap}steric_clash_fraction": (
                100
                * (min_cross_distances < 0.4).sum()
                / len(min_cross_distances)
                / args.num_predictions
            ).__round__(2),
            f"{overlap}self_intersect_fraction": (
                100
                * (min_self_distances < 0.4).sum()
                / len(min_self_distances)
                / args.num_predictions
            ).__round__(2),
            f"{overlap}mean_rmsd": rmsds[:, 0].mean(),
            f"{overlap}rmsds_below_2": (
                100 * (rmsds[:, 0] < 2).sum() / len(rmsds[:, 0])
            ),
            f"{overlap}rmsds_below_5": (
                100 * (rmsds[:, 0] < 5).sum() / len(rmsds[:, 0])
            ),
            f"{overlap}rmsds_percentile_25": np.percentile(rmsds[:, 0], 25).round(2),
            f"{overlap}rmsds_percentile_50": np.percentile(rmsds[:, 0], 50).round(2),
            f"{overlap}rmsds_percentile_75": np.percentile(rmsds[:, 0], 75).round(2),
            f"{overlap}mean_centroid": centroid_distances[:, 0].mean().__round__(2),
            f"{overlap}centroid_below_2": (
                100
                * (centroid_distances[:, 0] < 2).sum()
                / len(centroid_distances[:, 0])
            ).__round__(2),
            f"{overlap}centroid_below_5": (
                100
                * (centroid_distances[:, 0] < 5).sum()
                / len(centroid_distances[:, 0])
            ).__round__(2),
            f"{overlap}centroid_percentile_25": np.percentile(
                centroid_distances[:, 0], 25
            ).round(2),
            f"{overlap}centroid_percentile_50": np.percentile(
                centroid_distances[:, 0], 50
            ).round(2),
            f"{overlap}centroid_percentile_75": np.percentile(
                centroid_distances[:, 0], 75
            ).round(2),
        }
    )

    top5_rmsds = np.min(rmsds[:, :5], axis=1)
    top5_centroid_distances = centroid_distances[
        np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)
    ][:, 0]
    top5_min_cross_distances = min_cross_distances[
        np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)
    ][:, 0]
    top5_min_self_distances = min_self_distances[
        np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)
    ][:, 0]
    performance_metrics.update(
        {
            f"{overlap}top5_steric_clash_fraction": (
                100
                * (top5_min_cross_distances < 0.4).sum()
                / len(top5_min_cross_distances)
            ).__round__(2),
            f"{overlap}top5_self_intersect_fraction": (
                100
                * (top5_min_self_distances < 0.4).sum()
                / len(top5_min_self_distances)
            ).__round__(2),
            f"{overlap}top5_rmsds_below_2": (
                100 * (top5_rmsds < 2).sum() / len(top5_rmsds)
            ).__round__(2),
            f"{overlap}top5_rmsds_below_5": (
                100 * (top5_rmsds < 5).sum() / len(top5_rmsds)
            ).__round__(2),
            f"{overlap}top5_rmsds_percentile_25": np.percentile(top5_rmsds, 25).round(
                2
            ),
            f"{overlap}top5_rmsds_percentile_50": np.percentile(top5_rmsds, 50).round(
                2
            ),
            f"{overlap}top5_rmsds_percentile_75": np.percentile(top5_rmsds, 75).round(
                2
            ),
            f"{overlap}top5_centroid_below_2": (
                100 * (top5_centroid_distances < 2).sum() / len(top5_centroid_distances)
            ).__round__(2),
            f"{overlap}top5_centroid_below_5": (
                100 * (top5_centroid_distances < 5).sum() / len(top5_centroid_distances)
            ).__round__(2),
            f"{overlap}top5_centroid_percentile_25": np.percentile(
                top5_centroid_distances, 25
            ).round(2),
            f"{overlap}top5_centroid_percentile_50": np.percentile(
                top5_centroid_distances, 50
            ).round(2),
            f"{overlap}top5_centroid_percentile_75": np.percentile(
                top5_centroid_distances, 75
            ).round(2),
        }
    )

    top10_rmsds = np.min(rmsds[:, :10], axis=1)
    top10_centroid_distances = centroid_distances[
        np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)
    ][:, 0]
    top10_min_cross_distances = min_cross_distances[
        np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)
    ][:, 0]
    top10_min_self_distances = min_self_distances[
        np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)
    ][:, 0]
    performance_metrics.update(
        {
            f"{overlap}top10_self_intersect_fraction": (
                100
                * (top10_min_self_distances < 0.4).sum()
                / len(top10_min_self_distances)
            ).__round__(2),
            f"{overlap}top10_steric_clash_fraction": (
                100
                * (top10_min_cross_distances < 0.4).sum()
                / len(top10_min_cross_distances)
            ).__round__(2),
            f"{overlap}top10_rmsds_below_2": (
                100 * (top10_rmsds < 2).sum() / len(top10_rmsds)
            ).__round__(2),
            f"{overlap}top10_rmsds_below_5": (
                100 * (top10_rmsds < 5).sum() / len(top10_rmsds)
            ).__round__(2),
            f"{overlap}top10_rmsds_percentile_25": np.percentile(top10_rmsds, 25).round(
                2
            ),
            f"{overlap}top10_rmsds_percentile_50": np.percentile(top10_rmsds, 50).round(
                2
            ),
            f"{overlap}top10_rmsds_percentile_75": np.percentile(top10_rmsds, 75).round(
                2
            ),
            f"{overlap}top10_centroid_below_2": (
                100
                * (top10_centroid_distances < 2).sum()
                / len(top10_centroid_distances)
            ).__round__(2),
            f"{overlap}top10_centroid_below_5": (
                100
                * (top10_centroid_distances < 5).sum()
                / len(top10_centroid_distances)
            ).__round__(2),
            f"{overlap}top10_centroid_percentile_25": np.percentile(
                top10_centroid_distances, 25
            ).round(2),
            f"{overlap}top10_centroid_percentile_50": np.percentile(
                top10_centroid_distances, 50
            ).round(2),
            f"{overlap}top10_centroid_percentile_75": np.percentile(
                top10_centroid_distances, 75
            ).round(2),
        }
    )
for k in performance_metrics:
    print(k, performance_metrics[k])

if args.wandb:
    wandb.log(performance_metrics)
    histogram_metrics_list = [
        ("rmsd", rmsds[:, 0]),
        ("centroid_distance", centroid_distances[:, 0]),
        ("mean_rmsd", rmsds[:, 0]),
        ("mean_centroid_distance", centroid_distances[:, 0]),
    ]
    histogram_metrics_list.append(("top5_rmsds", top5_rmsds))
    histogram_metrics_list.append(("top5_centroid_distances", top5_centroid_distances))
    histogram_metrics_list.append(("top10_rmsds", top10_rmsds))
    histogram_metrics_list.append(
        ("top10_centroid_distances", top10_centroid_distances)
    )

    os.makedirs(f".plotly_cache/baseline_cache", exist_ok=True)
    images = []
    for metric_name, metric in histogram_metrics_list:
        d = {args.results_path: metric}
        df = pd.DataFrame(data=d)
        fig = px.ecdf(df, width=900, height=600, range_x=[0, 40])
        fig.add_vline(
            x=2,
            annotation_text="2 A;",
            annotation_font_size=20,
            annotation_position="top right",
            line_dash="dash",
            line_color="firebrick",
            annotation_font_color="firebrick",
        )
        fig.add_vline(
            x=5,
            annotation_text="5 A;",
            annotation_font_size=20,
            annotation_position="top right",
            line_dash="dash",
            line_color="green",
            annotation_font_color="green",
        )
        fig.update_xaxes(
            title=f"{metric_name} in Angstrom",
            title_font={"size": 20},
            tickfont={"size": 20},
        )
        fig.update_yaxes(
            title=f"Fraction of predictions with lower error",
            title_font={"size": 20},
            tickfont={"size": 20},
        )
        fig.update_layout(
            autosize=False,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend_title_text="Method",
            legend_title_font_size=17,
            legend=dict(
                yanchor="bottom",
                y=0.1,
                xanchor="right",
                x=0.99,
                font=dict(size=17),
            ),
        )
        fig.update_xaxes(showgrid=True, gridcolor="lightgrey")
        fig.update_yaxes(showgrid=True, gridcolor="lightgrey")

        fig.write_image(
            os.path.join(f".plotly_cache/baseline_cache", f"{metric_name}.png")
        )
        wandb.log(
            {
                metric_name: wandb.Image(
                    os.path.join(f".plotly_cache/baseline_cache", f"{metric_name}.png"),
                    caption=f"{metric_name}",
                )
            }
        )
        images.append(
            wandb.Image(
                os.path.join(f".plotly_cache/baseline_cache", f"{metric_name}.png"),
                caption=f"{metric_name}",
            )
        )
    wandb.log({"images": images})
