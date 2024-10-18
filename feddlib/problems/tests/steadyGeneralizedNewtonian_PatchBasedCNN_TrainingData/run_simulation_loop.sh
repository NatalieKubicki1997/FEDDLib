#!/bin/sh

#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM --mpi proc=4
#PJM -L elapse=0:05:00

#PJM -g pz0530

#PJM -j


source $HOME/Installation/load_env.sh 

# Loop over the mesh files
# Here manually change how many meshes one wants to consider 
for i in $(seq 1 2); do
    # Define mesh name based on iteration
    MESH_NAME="2D_artery_number_${i}.mesh"

    # Define output directory for this mesh run
    OUTPUT_DIR="results/2D_artery_number_mesh_${i}"

    # Create the output directory if it doesn't exist
    mkdir -p $OUTPUT_DIR

    # Generate a random velocity between 0.1 and 1.5
    MAX_VELOCITY=$(awk -v min=0.1 -v max=2.0 'BEGIN{srand(); printf "%.2f", min+rand()*(max-min)}')


    # Print out the random velocity for reference
    echo "Running simulation with MaxVelocity=$MAX_VELOCITY"

    # Modify the XML file to update the MaxVelocity value
    sed -i "s/<Parameter name=\"MaxVelocity\" type=\"double\" value=\"[^\"]*\"/<Parameter name=\"MaxVelocity\" type=\"double\" value=\"$MAX_VELOCITY\"/" parametersProblem.xml


    # Update the Mesh 1 Name in parametersProblem.xml
    # Using sed to replace the mesh entry in the XML file with the correct mesh name
    sed -i "s|<Parameter name=\"Mesh 1 Name\" type=\"string\" value=\"[^\"]*\"/>|<Parameter name=\"Mesh 1 Name\" type=\"string\" value=\"${MESH_NAME}\"/>|" parametersProblem.xml


    # Run the simulation and redirect output to the corresponding folder
    mpiexec ./problems_steadyGeneralizedNewtonian_PatchBasedCNN_TrainingData.exe > $OUTPUT_DIR/simulation.log 2>&1

    # Optional: Copy additional output files to the results directory if necessary
    # cp output_file_name $OUTPUT_DIR/
    cp *.xml $OUTPUT_DIR
    mv *.xmf *.h5 $OUTPUT_DIR
    echo "Completed simulation for $MESH_NAME, results stored in $OUTPUT_DIR"
done