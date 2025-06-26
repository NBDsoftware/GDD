from argparse import ArgumentParser, FileType, ArgumentTypeError
from utils.torsion import AngleCalcMethod

def parse_train_args():

    # General arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=FileType(mode="r"), default=None)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="workdir",
        help="Folder in which to save model and logs",
    )
    parser.add_argument(
        "--restart_dir",
        type=str,
        help="Folder of previous training model from which to restart",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="data/cache",
        help="Folder from where to load/restore cached dataset",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/PDBBind_processed/",
        help="Folder containing original structures",
    )
    parser.add_argument(
        "--split_train",
        type=str,
        default="data/splits/timesplit_no_lig_overlap_train",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--split_val",
        type=str,
        default="data/splits/timesplit_no_lig_overlap_val",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--split_test",
        type=str,
        default="data/splits/timesplit_test",
        help="Path of file defining the split",
    )
    parser.add_argument(
        "--test_sigma_intervals",
        action="store_true",
        default=False,
        help="Whether to log loss per noise interval",
    )
    parser.add_argument(
        "--val_inference_freq",
        type=int,
        default=5,
        help="Frequency of epochs for which to run expensive inference on val data",
    )
    parser.add_argument(
        "--train_inference_freq",
        type=int,
        default=None,
        help="Frequency of epochs for which to run expensive inference on train data",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps for inference on val",
    )
    parser.add_argument(
        "--num_inference_complexes",
        type=int,
        default=100,
        help="Number of complexes for which inference is run every val/train_inference_freq epochs (None will run it on all)",
    )
    parser.add_argument(
        "--inference_earlystop_metric",
        type=str,
        default="valinf_rmsds_lt2",
        help="This is the metric that is addionally used when val_inference_freq is not None",
    )
    parser.add_argument(
        "--inference_earlystop_goal",
        type=str,
        default="max",
        help="Whether to maximize or minimize metric",
    )
    parser.add_argument("--wandb", action="store_true", default=False, help="")
    parser.add_argument("--project", type=str, default="difdock_train", help="")
    parser.add_argument("--run_name", type=str, default="", help="")
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        default=False,
        help="CUDA optimization parameter for faster training",
    )
    parser.add_argument(
        "--num_dataloader_workers",
        type=int,
        default=0,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="pin_memory arg of dataloader",
    )

    # Training arguments
    parser.add_argument(
        "--n_epochs", type=int, default=400, help="Number of epochs for training"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--scheduler", type=str, default=None, help="LR scheduler")
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=20,
        help="Patience of the LR scheduler",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--restart_lr",
        type=float,
        default=None,
        help="If this is not none, the lr of the optimizer will be overwritten with this value when restarting from a checkpoint.",
    )
    parser.add_argument(
        "--w_decay", type=float, default=0.0, help="Weight decay added to loss"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for preprocessing"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="Whether or not to use ema for the model weights",
    )
    parser.add_argument(
        "--ema_rate",
        type=float,
        default=0.999,
        help="decay rate for the exponential moving average model parameters ",
    )

    # Dataset
    parser.add_argument(
        "--limit_complexes",
        type=int,
        default=0,
        help="If positive, the number of training and validation complexes is capped",
    )
    parser.add_argument(
        "--all_atoms",
        action="store_true",
        default=False,
        help="Whether to use the all atoms model",
    )
    parser.add_argument(
        "--receptor_radius",
        type=float,
        default=30,
        help="Cutoff on distances for receptor edges",
    )
    parser.add_argument(
        "--c_alpha_max_neighbors",
        type=int,
        default=10,
        help="Maximum number of neighbors for each residue",
    )
    parser.add_argument(
        "--atom_radius",
        type=float,
        default=5,
        help="Cutoff on distances for atom connections",
    )
    parser.add_argument(
        "--atom_max_neighbors",
        type=int,
        default=8,
        help="Maximum number of atom neighbours for receptor",
    )
    parser.add_argument(
        "--matching_popsize",
        type=int,
        default=20,
        help="Differential evolution popsize parameter in matching",
    )
    parser.add_argument(
        "--matching_maxiter",
        type=int,
        default=20,
        help="Differential evolution maxiter parameter in matching",
    )
    parser.add_argument(
        "--max_lig_size",
        type=int,
        default=None,
        help="Maximum number of heavy atoms in ligand",
    )
    parser.add_argument(
        "--remove_hs", action="store_true", default=False, help="remove Hs"
    )
    parser.add_argument(
        "--num_conformers",
        type=int,
        default=1,
        help="Number of conformers to match to each ligand",
    )
    parser.add_argument(
        "--esm_embeddings_path",
        type=str,
        default=None,
        help="If this is set then the LM embeddings at that path will be used for the receptor features",
    )

    # Diffusion
    parser.add_argument(
        "--tr_weight", type=float, default=0.33, help="Weight of translation loss"
    )
    parser.add_argument(
        "--rot_weight", type=float, default=0.33, help="Weight of rotation loss"
    )
    parser.add_argument(
        "--tor_weight", type=float, default=0.33, help="Weight of torsional loss"
    )
    parser.add_argument(
        "--rot_sigma_min",
        type=float,
        default=0.1,
        help="Minimum sigma for rotational component",
    )
    parser.add_argument(
        "--rot_sigma_max",
        type=float,
        default=1.65,
        help="Maximum sigma for rotational component",
    )
    parser.add_argument(
        "--tr_sigma_min",
        type=float,
        default=0.1,
        help="Minimum sigma for translational component",
    )
    parser.add_argument(
        "--tr_sigma_max",
        type=float,
        default=30,
        help="Maximum sigma for translational component",
    )
    parser.add_argument(
        "--tor_sigma_min",
        type=float,
        default=0.0314,
        help="Minimum sigma for torsional component",
    )
    parser.add_argument(
        "--tor_sigma_max",
        type=float,
        default=3.14,
        help="Maximum sigma for torsional component",
    )
    parser.add_argument(
        "--no_torsion",
        action="store_true",
        default=False,
        help="If set only rigid matching",
    )

    # Model
    parser.add_argument(
        "--num_conv_layers", type=int, default=2, help="Number of interaction layers"
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=5.0,
        help="Radius cutoff for geometric graph",
    )
    parser.add_argument(
        "--scale_by_sigma",
        action="store_true",
        default=True,
        help="Whether to normalise the score",
    )
    parser.add_argument(
        "--ns",
        type=int,
        default=16,
        help="Number of hidden features per node of order 0",
    )
    parser.add_argument(
        "--nv",
        type=int,
        default=4,
        help="Number of hidden features per node of order >0",
    )
    parser.add_argument(
        "--distance_embed_dim",
        type=int,
        default=32,
        help="Embedding size for the distance",
    )
    parser.add_argument(
        "--cross_distance_embed_dim",
        type=int,
        default=32,
        help="Embeddings size for the cross distance",
    )
    parser.add_argument(
        "--no_batch_norm",
        action="store_true",
        default=False,
        help="If set, it removes the batch norm",
    )
    parser.add_argument(
        "--use_second_order_repr",
        action="store_true",
        default=False,
        help="Whether to use only up to first order representations or also second",
    )
    parser.add_argument(
        "--cross_max_distance",
        type=float,
        default=80,
        help="Maximum cross distance in case not dynamic",
    )
    parser.add_argument(
        "--dynamic_max_cross",
        action="store_true",
        default=False,
        help="Whether to use the dynamic distance cutoff",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="MLP dropout")
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="sinusoidal",
        help="Type of diffusion time embedding",
    )
    parser.add_argument(
        "--sigma_embed_dim",
        type=int,
        default=32,
        help="Size of the embedding of the diffusion time",
    )
    parser.add_argument(
        "--embedding_scale",
        type=int,
        default=1000,
        help="Parameter of the diffusion time embedding",
    )

    args = parser.parse_args()
    return args



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def angle_calc_method_type(string):
    try:
        return AngleCalcMethod[string.upper()]
    except KeyError:
        raise ArgumentTypeError(f"Invalid value for AngleCalcMethod: {string}")



def parse_inference_args():

    parser = ArgumentParser()
    
    # IO ARGS #
    parser.add_argument(
        "--complex_name",
        type=str,
        default="1a0q",
        help="Name that the complex will be saved with",
    )
    parser.add_argument(
        "--protein_path", type=str, default=None, help="Path to the protein file"
    )
    parser.add_argument(
        "--ligand_description",
        type=str,
        default="CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1",
        help="Either a SMILES string or the path to a molecule file that rdkit can read",
    )
    parser.add_argument(
        "--protein_sequence",
        type=str,
        default=None,
        help="Sequence of the protein for ESMFold, this is ignored if --protein_path is not None",
    )
    parser.add_argument(
        "--protein_ligand_csv",
        type=str,
        default=None,
        help="Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters",
    )
    
    parser.add_argument(
        "--datadir",
        default="data/",
        help="""
            Path to the processed data with process_dataset.py
            
            The structure of thiis directory will be:
            
            datadir
            |-- XXXX
            |   |-- XXXX_ligand.mol2
            |   |-- XXXX_ligand.sdf
            |   |-- XXXX_protein_processed.pdb
            |   |-- rotXXXX_ligand.json
            |   |-- sphXXXX_ligand.csv
            |   |-- sphXXXX_ligand.json
            |   `-- torXXXX_ligand.json
            `-- data.csv
            
            Here, XXXX will usually be pdb ids, but it can be anything.
            """,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/user_inference",
        help="Directory where the outputs will be written to",
    )
    parser.add_argument(
        "--save_visualisation",
        action="store_true",
        default=False,
        help="Save a pdb file with all of the steps of the reverse diffusion",
    )
    parser.add_argument(
        "--save_inter_steps",
        default=False,
        help="Whether or not to save the results at intermediate intermediate steps",
    )
    
    # MODEL ARGS # 
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default="workdir/paper_score_model",
        help="Path to folder with trained score model and hyperparameters",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="best_ema_inference_epoch_model.pt",
        help="Checkpoint to use for the score model",
    )
    parser.add_argument(
        "--confidence_model_dir",
        type=str,
        default="workdir/paper_confidence_model",
        help="Path to folder with trained confidence model and hyperparameters",
    )
    parser.add_argument(
        "--confidence_ckpt",
        type=str,
        default="best_model_epoch75.pt",
        help="Checkpoint to use for the confidence model",
    )
    
    # SAMPLING ARGS #
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument(
        "--samples_per_complex", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--inference_steps", type=int, default=20, help="Number of denoising steps"
    )
    parser.add_argument(
        "--no_final_step_noise",
        action="store_true",
        default=False,
        help="Use no noise in the final step of the reverse diffusion",
    )
    parser.add_argument(
        "--actual_steps",
        type=int,
        default=None,
        help="Number of denoising steps that are actually performed",
    )


    ## GUIDANCE ARGS ##

    # General parameters #
    parser.add_argument(
        "--gamma_scheduler", 
        type=str, 
        required=False, 
        default="constant",
        help="""
            Scheduler to use for gamma.
            """
    )
    parser.add_argument(
        "--dynamic_gamma",
        action="store_true",
        default=False,
        help="""
            Computes gamma dynamically at each diffusion step based on distance 
            to the desired regions and the cosine similarity between the geodesic
            and the current update
        """,
    )
    parser.add_argument(
        "--update_regions",
        default=False,
        help="""
            Wether or not to update regions for the eta parameter ablation.
            Makes sense when the input is obtained by process_dataset.py, where
            a single \eta is used for all values and it might be useful
            to update them.
        """
    )

    # Translation #
    parser.add_argument(
        "--translation_guiding",
        action="store_true",
        default=False,
        help="Applies guiding to translational diffusion",
    )
    parser.add_argument("--tr_gamma", type=float, default=0.3, help="Guidance strength")
    parser.add_argument(
        "--tr_margin",
        type=float,
        required=False,
        default=None,
        help="""
            Margin used for sphere radius. This parameter was added to override
            the margin used in the preprocessing.
        """,
    )
    parser.add_argument(
        "--predict_pocket",
        action="store_true",
        required=False,
        default=None,
        help="""
            Use pocket prediction for guidance.
        """
    )
    parser.add_argument(
        "--predict_pocket_method",
        type=str, # fpocket / p2rank
        required=False,
        default=None,
        help="Method to use to predict the pocket. Requires them to be installed in ./tools/"
    )

    # Rotation #
    parser.add_argument(
        "--rotation_guiding",
        action="store_true",
        default=False,
        help="Applies guiding to rotational diffusion",
    )
    parser.add_argument("--rot_gamma", type=float, default=0.3, help="Guidance strength")
    parser.add_argument(
        "--rot_margin",
        type=float,
        required=False,
        default=None,
        help="""
            Margin used for rot angles. This parameter was added to override
            the margin used in the preprocessing.
        """,
    )

    # Torsion #
    parser.add_argument(
        "--torsion_guiding",
        action="store_true",
        default=False,
        help="Applies guiding to torsional diffusion",
    )
    parser.add_argument("--tor_gamma", type=float, default=0.3, help="Guidance strength")
    parser.add_argument(
        "--tor_margin",
        type=float,
        required=False,
        default=None,
        help="""
            Margin used for torsion angles. This parameter was added to override
            the margin used in the preprocessing.
        """,
    )

    parser.add_argument(
        "--mask_n_tor", 
        type=int, 
        default=0, 
        help="""
            Mask n rotatable bonds for torsional guidance.
            
            This has two purposes:
                1. Add regularization by relaxing the constraints imposed on the angles
                2. When doing angle transfer for MCS docking, sometimes its necesary to provide
                   more than one region (although only one is correct). Because GeoDirDock always
                   guides to the closest region, it is possible that due the the random
                   sampling at the beginning of the inference, the closest region is the
                   incorrect one. By turning off guidance for some angles, we let the rotatable
                   bond naturally get closer to the correct region.
                   
                   Using this showed improvements in the MCS docking.
        """
    )
    parser.add_argument(
        "--mask_n_distribution",
        type=str,
        default="random",
        help="""
            Distribution to mask the n random angles. It can be 'random' or 'weighted'
            
            weighted distribution is given by the circular std.
            more precisely, a number of conformers are generated with
            rdkit, and those with a lower circular std are assumed to be
            less likely wrong, and are assigned lower probabilities to be
            masked.

            this assumption should be taken with a grain of salt, but it also showed improvements
        """,
    )
    
    parser.add_argument(
        "--n_regions",
        required=False,
        default=None,
        help="""
        Number of regions to use. This was added for a benchmark, but in practice it should
        probably just be the number of regions provided, leave @ default for now
        """,
    )

    ## Note:
    # all the *_margin arguments were added to override the values set in preprocessing
    # to benchmark these.
    # They are updated in a quick and dirty way by updating the regions using the difference
    # between the provided argument and the default margins in the preprocessing (7, 0.15 and 0.15) for tr, rot, tor

    # Not providing these is the same as setting them to 0


    ### args for the becnhmark: all of these can be harcoded to their defaults ###
    parser.add_argument(
        "--reordering",
        type=str2bool,
        required=False,
        default=True,
        help="Whether the ligand and ground_ligand were ordered or not",
    )
    parser.add_argument(
        "--neg_vdir",
        type=str2bool,
        required=False,
        default=True,
        help="Use a negative vdir for torsion or not",
    )
    parser.add_argument(
        "--add_hs_conformer",
        type=str2bool,
        required=False,
        default=True,
        help="Whether Hs are removed before generating the conformer or not",
    )
    parser.add_argument(
        "--remove_hs_mol",
        type=str2bool,
        required=False,
        default=True,
        help="Whether Hs are removed in complex_graph.mol",
    )
    parser.add_argument("--reorder_transfer", required=False, default=None)
    parser.add_argument(
        "--angle_calc_method",
        type=angle_calc_method_type,
        required=False,
        help="""
                        Method used to calculate the torsion angles.
                        Torsion angles are defined by 4 atoms: a--(b--c)--d, where 
                        b and c form the rotatable bond and a and d are neighbor atoms
                        to b and c, respectively
                        
                        tor_calc_1: Uses all neighbors a_i and d_j to compute the angle
                        tor_calc_2: Uses single neighbors a and d to compute the angle
                        """,
        default="tor_calc_2"
    )
    parser.add_argument(
        "--Rm_update_method",
        required=False,
        default="m0",
        help="""
                        Method to compute the guided updates in Rm. Valid for both translation and torsion
                        
                        m0: Original method
                            updates = (1 - gamma) * update_dir * update_dist  + gamma * vdir * distance
                            
                        m1: Original method, with no distance
                            updates = (1 - gamma) * update_dir * update_dist  + gamma * vdir
                            
                        m2: Guide only direction and use the distance vector magnitude:
                            Less stability when sigma is lower (higher gammas yielded worse results),
                            but stronger guidance.
                            
                            updates = ((1 - gamma) * update_direction + gamma * vdir ) * distance
                            
                        m3: Guide only direction and use update vector magnitude: 
                            Good for stability when sigma is lower, but less guidance
            
                            updates = ((1 - gamma) * update_direction + gamma * vdir ) * update_magnitude
                            
                        m4: Guide distance and direction independently. Allows for different scalars.

                            guided_direction = (1 - gamma) * update_direction + gamma * vdir
                            guided_distance = (1 - gamma) * update_magnitude + gamma * distance
                            
                            updates = guided_direction * guided_distance
                            
                        """,
    )
    
    
    args = parser.parse_args()
    return args