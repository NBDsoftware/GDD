import os
import shutil

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Parse folder name
        parts = folder.split('_')
        protein_name = parts[0]  # First part is always protein
        ligand_name = parts[2]   # Third part is always ligand
        
        # Create new folder name and path
        new_folder_name = f"{protein_name}_{ligand_name}"
        new_folder_path = os.path.join(output_dir, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        
        # Process files in the folder
        for file in os.listdir(folder_path):
            if file.startswith("."): 
                continue
            file_path = os.path.join(folder_path, file)
            # Handle protein file
            if 'protein_processed.pdb' in file:
                new_name = f"{protein_name}_protein_processed.pdb"
                shutil.copy2(file_path, os.path.join(new_folder_path, new_name))
            # Handle ligand file
            elif 'ligand.pdb' in file:
                # Simply match the folder name prefix for ligand
                if folder in file:  # This ensures we get the right numbered version
                    new_name = f"{ligand_name}.pdb"
                    shutil.copy2(file_path, os.path.join(new_folder_path, new_name))

# Usage
if __name__ == "__main__":
    import sys
   
    if len(sys.argv) != 2:
        print("Usage: python prepare_dockgen.py <directory_path>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = input_directory + "_processed"
    
    process_directory(input_directory, output_directory)
