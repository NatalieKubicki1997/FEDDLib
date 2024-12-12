import os
import re

def check_nonlinear_iterations(folder_path, target_iterations=10):
    # Updated pattern to match "Total nonlinear iterations : x" even with extra characters around it
    pattern = re.compile(r"### Total nonlinear iterations\s*:\s*(\d+)")
    
    # List to store matching file paths
    matching_files = []
    
    # Walk through all subdirectories and files within folder_path
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file == "simulation.log":
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    for line in f:
                        # Search for the pattern in each line
                        match = pattern.search(line)
                        if match:
                            # Extract the number of iterations
                            iterations = int(match.group(1))
                            if iterations == target_iterations:
                                print(f"Found target iterations ({target_iterations}) in file: {file_path}")
                                matching_files.append(file_path)
                                break  # Stop reading further lines in this file
    
    if not matching_files:
        print(f"No files found with {target_iterations} nonlinear iterations.")
    else:
        print(f"\nTotal files with {target_iterations} nonlinear iterations: {len(matching_files)}")
    
    return matching_files

# Example usage
folder_path = "simulation_results"  # Replace with the actual path to folder A
check_nonlinear_iterations(folder_path)
