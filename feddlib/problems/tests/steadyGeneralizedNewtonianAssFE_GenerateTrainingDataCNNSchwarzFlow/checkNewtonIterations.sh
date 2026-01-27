import os
import re

def check_nonlinear_iterations(folder_path): 
    # Der spezifische Fehlerstring aus deinem CFD-Solver
    error_pattern = "Throw test that evaluated to true: nlIts == maxNonLinIts"
    
    matching_files = []
    
    print(f"Scanning folder: {folder_path} for non-converged simulations...\n")

    # Gehe durch alle Unterverzeichnisse
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file == "simulation.log":
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r') as f:
                    # Wir lesen die Datei, um nach dem Fehler zu suchen
                    content = f.read()
                    if error_pattern in content:
                        # Extrahiere den Ordnernamen für eine bessere Übersicht
                        folder_name = os.path.basename(root)
                        print(f"[!] Fehler gefunden in: {folder_name} ({file_path})")
                        matching_files.append(file_path)
    
    # Zusammenfassung
    if not matching_files:
        print("-" * 30)
        print("Sauber! Keine Dateien mit 'maxNonLinIts' Fehlern gefunden.")
    else:
        print("-" * 30)
        print(f"Gefunden: {len(matching_files)} Simulationen sind nicht konvergiert.")
    
    return matching_files

# Beispielaufruf
folder_path = "simulation_results" 
failed_sims = check_nonlinear_iterations(folder_path)