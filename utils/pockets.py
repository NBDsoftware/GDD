import subprocess
import numpy as np
import os
import shutil
import ipdb
import pandas as pd


def get_pocket_center_fpocket(protein_filepath):
    """"""
    # run fpocket
    subprocess.run(['fpocket', '-f', protein_filepath], check=True)
    protein_name, _ = os.path.splitext(protein_filepath)
    output_dir = f"{protein_name}_out"
    try:
        # Parse pocket1_atm.pdb coordinates
        coords = []
        with open(os.path.join(output_dir, 'pockets', 'pocket1_atm.pdb')) as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
        
        # Calculate centroid
        center = np.round(np.mean(coords, axis=0), 3)
        
        return center
    
    finally:
        # Cleanup fpocket output
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

def get_pocket_center_p2rank(pdb_file):
    def _find_project_root():
        current = os.path.dirname(os.path.abspath(__file__))
        while current != '/':  # Stop at root directory
            if os.path.exists(os.path.join(current, 'tools', 'p2rank_2.5')):
                return current
            current = os.path.dirname(current)
        raise FileNotFoundError("Could not find project root (no tools/p2rank_2.5 directory found)")

    # Find p2rank executable
    project_root = _find_project_root()
    p2rank_path = os.path.join(project_root, "tools", "p2rank_2.5", "prank")
    
    if not os.path.exists(p2rank_path):
        raise FileNotFoundError(f"P2Rank executable not found at: {p2rank_path}")
        
    pdb_file = os.path.abspath(pdb_file)
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    # Create temporary directory for output
    pdb_dir = os.path.dirname(pdb_file)
    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
    temp_dir = os.path.join(pdb_dir, f"p2rank_temp_{pdb_name}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        subprocess.run(
            [
                p2rank_path,
                'predict',
                '-f',
                pdb_file,
                '-o', temp_dir
            ],
            check=True,
        )
        
        # Find predictions CSV file
        csv_path = None
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('_predictions.csv'):
                    csv_path = os.path.join(root, file)
                    break
        
        if csv_path is None:
            raise ValueError("CSV not found")
        
        df = pd.read_csv(csv_path)
        
        # Strip any remaining whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Get coordinates of top pocket (first row)
        center = np.array([
            float(str(df.iloc[0]['center_x']).strip()),
            float(str(df.iloc[0]['center_y']).strip()),
            float(str(df.iloc[0]['center_z']).strip())
        ])
        
        return np.round(center, 3)
    
    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
def main():
    protein_path = "data/1a0q/1a0q_protein_processed.pdb"
    # pocket = get_pocket_center_fpocket(protein_filepath=protein_path)
    pocket = get_pocket_center_p2rank(protein_path)
    print(f"{pocket=}")

if __name__ == "__main__":
    main()
    