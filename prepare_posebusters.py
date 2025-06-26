import os
import shutil

def cleanup_directories(parent_dir):
    """
    Clean up multiple directories containing protein and ligand files.
    
    Args:
        parent_dir (str): Path to the parent directory containing subdirectories
    """
    try:
        # Get all subdirectories
        subdirs = [d for d in os.listdir(parent_dir) 
                  if os.path.isdir(os.path.join(parent_dir, d))]
        
        for subdir in subdirs:
            try:
                # Split directory name to get components (e.g., "5S8I_2LY" -> "5S8I", "2LY")
                components = subdir.split('_')
                if len(components) != 2:
                    print(f"Skipping {subdir}: not in expected format (e.g., 5S8I_2LY)")
                    continue
                    
                pdb_id, ligand_code = components
                subdir_path = os.path.join(parent_dir, subdir)
                
                # Process each file in the subdirectory
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    if not os.path.isfile(file_path):
                        continue
                        
                    # Handle protein file
                    if "protein.pdb" in filename:
                        new_name = f"{pdb_id}_protein.pdb"
                        new_path = os.path.join(subdir_path, new_name)
                        if new_path != file_path:
                            shutil.move(file_path, new_path)
                            print(f"{subdir}: Renamed {filename} to {new_name}")
                    
                    # Handle main ligand file
                    elif f"{subdir}_ligand.sdf" == filename:
                        new_name = f"{ligand_code}.sdf"
                        new_path = os.path.join(subdir_path, new_name)
                        if new_path != file_path:
                            shutil.move(file_path, new_path)
                            print(f"{subdir}: Renamed {filename} to {new_name}")
                
                print(f"Processed directory: {subdir}")
                
            except Exception as e:
                print(f"Error processing directory {subdir}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error accessing parent directory: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python prepare_posebusters.py <posebusters_directory_path>")
        sys.exit(1)
        
    cleanup_directories(sys.argv[1])