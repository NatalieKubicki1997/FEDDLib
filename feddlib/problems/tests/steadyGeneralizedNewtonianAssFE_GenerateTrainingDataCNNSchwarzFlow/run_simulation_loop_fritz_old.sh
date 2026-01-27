#!/bin/bash -l

#SBATCH -N 1
#SBATCH --ntasks=36
#SBATCH -t 00:10:00
#SBATCH --output=multi_sim.out
#SBATCH --error=multi_sim.err
#SBATCH --switches=1

# --- Setup Environment ---
source ~/Installation/load_updated_intel.sh
export OMP_NUM_THREADS=1

# --- Paths and Files ---
MESH_PATH="/home/hpc/k105be/k105be12/Run_Folder_FEDDLib/Generate_TrainingData_CNN-Schwarz-Flow/NewMeshes/meshFiles"
CSV_FILE="velocity_data_new.csv"
EXE="./problems_steadyGeneralizedNewtonianAssFE_GenerateTrainingDataCNNSchwarzFlow.exe"

# --- Loop over simulations (0 to 800 based on your list) ---
#for i in $(seq 968 1044); do
#for i in 918 813 818 888 892 929 809 928 890 870 934 1017 809 924; do
for i in 813; do
    # 1. Extract Velocity from CSV
    # We look for the line starting with "i," and take the 2nd column
    MAX_VELOCITY=$(awk -F',' -v id="$i" '$1 == id {print $2}' "$CSV_FILE")

    # 2. Define Mesh Name
    # Note: Check if the filename pattern (maxdiameter_0_001_totallength_0_02 vs 0_024) matches exactly!
    MESH_NAME="${MESH_PATH}/2D_stenotic_artery_mesh_maxdiameter_0_001_totallength_0_024_number_${i}.mesh"
    
    MESH_BASENAME=$(basename "$MESH_NAME" .mesh)
    OUTPUT_DIR="simulation_results/${MESH_BASENAME}"

    echo "--- Starting Sim $i | Velocity: $MAX_VELOCITY | Mesh: $MESH_BASENAME ---"

    # 3. Prepare Directory
    mkdir -p "$OUTPUT_DIR/input"
    mkdir -p "$OUTPUT_DIR/output"

    # Skip if velocity is 0.00 (as seen in your list) to save time, or if empty -> Zeros are written out its okey
    #if [[ -z "$MAX_VELOCITY" || "$MAX_VELOCITY" == "0.00" ]]; then
    #    echo "Skipping Sim $i: Velocity is $MAX_VELOCITY"
    #    continue
    #fi

    # 4. Modify XML (In-place)
    # Update MaxVelocity
    sed -i "s/<Parameter name=\"MaxVelocity\" type=\"double\" value=\"[^\"]*\"/<Parameter name=\"MaxVelocity\" type=\"double\" value=\"$MAX_VELOCITY\"/" parametersProblem_PicardNewton.xml
    
    # Update Mesh Path
    sed -i "s|<Parameter name=\"Mesh 1 Name\" type=\"string\" value=\"[^\"]*\"/>|<Parameter name=\"Mesh 1 Name\" type=\"string\" value=\"${MESH_NAME}\"/>|" parametersProblem_PicardNewton.xml

    # 5. Run Simulation
    # srun uses the 36 tasks defined in the SBATCH header
    srun $EXE > "$OUTPUT_DIR/simulation.log" 2>&1

    # 6. Cleanup / Move Data
    # Move Flags.xmf to the input folder
    mv Flags.xmf Flags.h5 "$OUTPUT_DIR/input/"

    # Move velocity.xmf to the output folder
    mv velocity.xmf velocity.h5 viscosity.h5 viscosity.xmf pressure.xmf pressure.h5 "$OUTPUT_DIR/output/"

    # Optionally: Copy additional output files to the results directory if necessary
    cp *.xml "$OUTPUT_DIR/"

    echo "Completed simulation for $MESH_NAME, results stored in $OUTPUT_DIR"

    echo "Finished Sim $i"
done
