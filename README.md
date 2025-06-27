# GeoDirDock: Guiding Docking Along Geodesic Paths

[![DOI](https://zenodo.org/badge/1005539997.svg)](https://doi.org/10.5281/zenodo.15755563)


## Installing GeoDirDock

To install GeoDirDock, first create a conda environment:

```
conda env create --file environment_gdd.yml
conda activate geodirdock
```

Need `torch` to install `openfold`, so do it now:

```
pip install "openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307"
```

*Note: Depending on your configuration, you might need to also export the conda
environment library path:

```
export LD_LIBRARY_PATH=/path/to/anaconda/envs/geodirdock/lib:$LD_LIBRARY_PATH
```

## Running GeoDirDock

To run GeoDirDock, a csv with the following format is required:

```
complex_name,protein_path,ligand_description,protein_sequence
```

For example:

```
complex_name,protein_path,ligand_description,protein_sequence
6ic0,data2/6ic0/6ic0_protein_processed.pdb,O=C(c1ccncc1Nc1cc2nccnc2c(c1)c1ccc2c(c1)n(C)cc2)Nc1cncnc1,
```

The directory specificied in the `protein_path` column expects the following structure:

```
.
|-- complex
|   |-- ligand_name.ext -> ligand file, if available
|   |-- {protein_name}_protein_processed.pdb -> protein file
|   |-- rot{ligand_name}.json -> defined docking regions for rotation
|   |-- sph{ligand_name}.csv -> defined docking regions for translation
|   `-- tor{ligand_name}.json -> defined docking regions for torsion
`-- data.csv
```

Here, `complex` can be either be a single PDB, in which case the name of the ligand will be set to `PDB_ligand` or a `PDB_Y`, where `Y` is the name of the ligand. 

The files containing the defined docking regions should have the following structure:

### Translation

A csv file like this:

```
x,y,z,R
```

Where `x, y, z` are the center of the docking sphere and `R` is the radius

### Rotation

A json file like this:

```
{
    [phi_1-eta, phi_1+eta],
    [psi_1-eta, psi_1+eta],
    [phi_2-eta, phi_2+eta],
    [psi_2-eta, psi_2+eta],
}
```

Where `phi_1`, `psi_1`, `phi_2` and `psi_2` are the angles described in the paper and `eta` controls the size of the regions.

### Torsions

For N torsion angles and K regions for each one, a json like:

```
{
    [t_0^0-eta, t_0^0+eta, ..., t_0^K-eta, t_0^K+eta],
    [...],
    [t_N^0-eta, t_N^0+eta, ..., t_N^K-eta, t_N^K+eta]
}

```

Each parameter defining a region can be manually provided or obtained with `process_dataset.py`.

## Example

```
CSV_PATH="./data2/testset_subset_6ic0.csv"
DATA_PATH="./data2"
OUTPUT_PATH="./results/"

GAMMA=0.3

python -m inference --protein_ligand_csv $CSV_PATH --out_dir $OUTPUT_PATH --datadir $DATA_PATH \
  --inference_steps 20 --samples_per_complex 40 --batch_size 4 --actual_steps 18 --no_final_step_noise \
  --translation_guiding --tr_gamma $GAMMA \
  --torsion_guiding --tor_gamma $GAMMA \
  --rotation_guiding --rot_gamma $GAMMA  \
  --tr_margin 15 --rot_margin 0.2 --tor_margin 0.35 
```

You can run this example by doing `./run_inference.sh`

The arguments `--tr_margin`, `--rot_margin` and `tor_margin` refer to `R` and `η` in the paper, while `gamma` controls the guidance strength; and all can be freely set. Guidance for specific subspaces can be included or excluded with the respective flags.

## Running GeoDirDock on large scale datasets for self-docking

To evaluate GDD on large scale dataset, one can obtained the desired regions with `process_dataset.py`.

`process_dataset.py` can take in a directory of `pdb` files structured as follows:

```
dataset/
|-- xxxx.pdb
|-- xxxx.pdb
|-- xxxx.pdb
|-- ...
```

And will produce a directory with the same structure as described above:

```
dataset_processed/
|-- complex
|   |-- {ligand_name}.pdb -> ligand files, extracted from the initial PDB
|   |-- {protein_name}_protein_processed.pdb -> protein file
|   |-- rot{ligand_name}.json -> defined docking regions for rotation
|   |-- sph{ligand_name}.csv -> defined docking regions for translation
|   `-- tor{ligand_name}.json -> defined docking regions for torsion
`-- data.csv
```

Here, ligands are extracted from the protein files (for more details see the `process_dataset.py` script), and the `data.csv` file is created. This `data.csv` can be used as input to run inference in the dataset:

```
python process_dataset.py --data_directory path/to/directory_with_pdbs \
--get_ground_ligand_pose
```

Alternatively, if specific files for the protein and ligands are available, one can also use the `--structured_data_direcory` flag. This will skip the first step; only processing proteins, obtaining the regions and creating the `data.csv` file necessary for inference.

```
python process_dataset.py --data_directory path/to/structured_directory \
--structured_directory \
--get_ground_ligand_pose
```

In this case, one might want to specify the given ligand extension using the `ligand_ext` argument. By default, when taking a directory of raw pdbs this will be `pdb`, and `sdf` otherwise:

```
python process_dataset.py --data_directory path/to/structured_directory \
--structured_directory \
--get_ground_ligand_pose \
--ligand_ext ext
```

In the case of already having processed proteins, or not wanting to process them, the argument `skip_protein_processing` can be used:

```
python process_dataset.py --data_directory path/to/structured_directory_processed \
--structured_directory \
--skip_protein_processing \
--get_ground_ligand_pose
```

Finally, processed datasets are stored in a new directory named `args.data_directory` + `_processed` by default. To avoid creating a new directory, (e.g. the input directory is already structured and processed), the flag `--in_place` can be used to avoid so:

```
python process_dataset.py --data_directory path/to/structured_directory_processed \
--structured_directory \
--skip_protein_processing \
--in_place \
--get_ground_ligand_pose
```

### Running GeoDirDock on the PDBBind test set

To run GeoDirDock on PDBBind, follow the instructions outlined in the [DiffDock GitHub Repository](https://github.com/gcorso/DiffDock) to download the PDBBind dataset. Then, optionally create the `PDBBind_processed_testset` by filtering using the splits in `data`.

Because this directory is already structured and processed, we run:

```
python process_dataset.py --data_directory data/PDBBind_processed_testset \
--structured_directory \
--skip_protein_processing \
--get_ground_ligand_pose \

# --in_place (optionally, if not included the PDBBind_processed_testset_processed directory will be created)
```

### Running GeoDirDock on the PoseBusters Benchmark Set

To download this dataset, download it from the source given in the [PoseBusters paper](https://arxiv.org/pdf/2308.05777) and place the `posebusters_benchmark_set` under `data/`

The structure of the downloaded dataset is the following:

```
5S8I_2LY
├── 5S8I_2LY_ligand.sdf
├── 5S8I_2LY_ligands.sdf
├── 5S8I_2LY_ligand_start_conf.sdf
└── 5S8I_protein.pdb

```
As such, we need to modify it slightly to adapt it to the expected input directory structure. To do so, run the `prepare_posebusters.py` script:

```
python prepare_posebusters.py path/to/posebusters_benchmark_set
```

With the correct structure, we now can run the process script similarly:

```
python process_dataset.py --data_directory data/posebusters_benchmark_testset \
 --structured_directory \
  --get_ground_ligand_pose 

# --in_place (optionally, if not included the posebusters_benchmark_set_processed directory will be created)
# --skip_protein_processing (in this case the proteins aren't processed, so we do not include the flag)
```

### Running GeoDirDock on the DockGen test set

To run GeoDirDock on DockGen, follow the instructions outlined in the [DiffDock GitHub Repository](https://github.com/gcorso/DiffDock) to download the DockGen dataset. Then, optionally create the `DockGen_testset` by filtering using the splits provided therein.

The structure of the downloaded dataset is the following:
```
3ju4_1_SLB_2/
├── 3ju4_1_SLB_0_ligand.pdb
├── 3ju4_1_SLB_1_ligand.pdb
├── 3ju4_1_SLB_2_ligand.pdb
└── 3ju4_1_SLB_2_protein_processed.pdb
```

Similarly to PoseBusters, we need an additional step to adapt the dataset. In this case, we only keep the main ligand (i.e. the one that shared prefix with the folder and protein). This will create the `DockGen_testset_processed` directory (assuming it was created by filtering as noted above):

```
python prepare_dockgen.py path/to/DockGen_testset
```

Finally, we process the dataset with:

```
python process_dataset.py --data_directory data/DockGen_testset_processed \
--structured_directory \
--skip_protein_processing \ #  (already processed)
--get_ground_ligand_pose \
--ligand_ext pdb # unlike PoseBusters, ligands file are pdb files

# --in_place (optionally, if not included the DockGen_testset_processed directory will be created)
```
For all datasets, inferences can be run the same way by using the appropiate paths:

```
CSV_PATH="./path/to/dataset/data.csv"
DATA_PATH="./path/to/dataset/
OUTPUT_PATH="./results/"

GAMMA=0.3

python inference.py --protein_ligand_csv $CSV_PATH --out_dir $OUTPUT_PATH --datadir $DATA_PATH \
  --inference_steps 20 --samples_per_complex 40 --batch_size 4 --actual_steps 18 --no_final_step_noise \
  --translation_guiding --tr_gamma $GAMMA \
  --torsion_guiding --tor_gamma $GAMMA \
  --rotation_guiding --rot_gamma $GAMMA  \
  --tr_margin 15 --rot_margin 0.2 --tor_margin 0.35 
```

## Computing Metrics

To compute metrics on docking results, one can use the `read_results.py` script:

```
RESULTS_DIR="./path/to/results"
DATA="./path/to/dataset"
CSV="./path/to/dataset/data.csv"


python read_results.py --protein_ligand_csv "$CSV" \
    --results_path "$RESULTS_DIR" \
    --datadir "$DATA" \
    --num_predictions N  \
    --save_readings \
    --ligand_ext ext \

#   --inter_steps
```

This script needs how many predictions were sampled in the docking process (`--num_predictions`) and optionally,
can compute metrics for intermediate steps with the `--inter_steps` argument. To do so, one must also have provided the `--save_inter_steps` in the inference script (off by default and only useful for benchmarking purposes). The ligand extension of the ground truth ligands must also provided with `--ligand_ext`.


