#!/bin/sh

#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM --mpi proc=4
#PJM -L elapse=0:05:00

#PJM -g pz0530

#PJM -j


source $HOME/Installation/load_env.sh 

# Loop over the mesh files
# Manually change the range to control how many meshes you want to process
for i in $(seq 0 20); do
    # Define the mesh path
    MESH_PATH="/work/pz0530/z30530/Japan_PatchBasedCNN_BloodFlow/mesh_generation/meshFiles"

    # Define mesh name based on iteration, including the path
    MESH_NAME="${MESH_PATH}/2D_stenotic_artery_mesh_maxdiameter_0_001_totallength_0_02_number_${i}.mesh"

    # Define output directory based on the mesh name without the .mesh extension
    MESH_BASENAME=$(basename "$MESH_NAME" .mesh)
    OUTPUT_DIR="simulation_results/${MESH_BASENAME}"

    # Create the output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    # Generate a random velocity between 0.1 and 1.5
    MAX_VELOCITY=$(awk -v min=0.1 -v max=0.5 'BEGIN{srand(); printf "%.2f", min+rand()*(max-min)}')

    # Print out the random velocity for reference
    echo "Simulation Run ${i}: Running simulation with MaxVelocity=$MAX_VELOCITY"

    # Modify the XML file to update the MaxVelocity value
    sed -i "s/<Parameter name=\"MaxVelocity\" type=\"double\" value=\"[^\"]*\"/<Parameter name=\"MaxVelocity\" type=\"double\" value=\"$MAX_VELOCITY\"/" parametersProblem.xml

    # Update the Mesh 1 Name in parametersProblem.xml
    # Using sed to replace the mesh entry in the XML file with the correct mesh name
    sed -i "s|<Parameter name=\"Mesh 1 Name\" type=\"string\" value=\"[^\"]*\"/>|<Parameter name=\"Mesh 1 Name\" type=\"string\" value=\"${MESH_NAME}\"/>|" parametersProblem.xml

    # Run the simulation and redirect output to the corresponding folder
    mpiexec ./problems_steadyGeneralizedNewtonian_PatchBasedCNN_TrainingData.exe > "$OUTPUT_DIR/simulation.log" 2>&1

    # Create input and output directories within the output directory
    mkdir -p "$OUTPUT_DIR/input"
    mkdir -p "$OUTPUT_DIR/output"

    # Move Flags.xmf to the input folder
    mv Flags.xmf Flags.h5 "$OUTPUT_DIR/input/"

    # Move velocity.xmf to the output folder
    mv velocity.xmf velocity.h5 "$OUTPUT_DIR/output/"

    # Optionally: Copy additional output files to the results directory if necessary
    cp *.xml "$OUTPUT_DIR/"

    echo "Completed simulation for $MESH_NAME, results stored in $OUTPUT_DIR"
done