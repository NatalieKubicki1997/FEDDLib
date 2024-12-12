#!/bin/bash

# Define the main directory containing the simulation results
SIMULATION_DIR="/work/pz0530/z30530/Japan_PatchBasedCNN_BloodFlow/simulation/build_feddlib/feddlib/problems/tests/steadyGeneralizedNewtonian_PatchBasedCNN_TrainingData/simulation_results"

# Define the required files for input and output directories
INPUT_FILES=("Flags.h5" "Flags.xmf")
OUTPUT_FILES=("pressure.h5" "pressure.xmf" "velocity.h5" "velocity.xmf" "viscosity.h5" "viscosity.xmf")

# Iterate through each subdirectory in the simulation results directory
for SUBDIR in "$SIMULATION_DIR"/*; do
    if [ -d "$SUBDIR" ]; then
        echo "Checking folder: $(basename "$SUBDIR")"
        
        # Check the input directory
        INPUT_DIR="$SUBDIR/input"
        MISSING_INPUT_FILES=()
        for FILE in "${INPUT_FILES[@]}"; do
            if [ ! -f "$INPUT_DIR/$FILE" ]; then
                MISSING_INPUT_FILES+=("$FILE")
            fi
        done

        # Check the output directory
        OUTPUT_DIR="$SUBDIR/output"
        MISSING_OUTPUT_FILES=()
        for FILE in "${OUTPUT_FILES[@]}"; do
            if [ ! -f "$OUTPUT_DIR/$FILE" ]; then
                MISSING_OUTPUT_FILES+=("$FILE")
            fi
        done

        # Report missing files or confirm that all are present
        if [ ${#MISSING_INPUT_FILES[@]} -eq 0 ] && [ ${#MISSING_OUTPUT_FILES[@]} -eq 0 ]; then
            echo "All required files are present in 'input' and 'output' directories."
        else
            echo "Missing files**************:"
            if [ ${#MISSING_INPUT_FILES[@]} -ne 0 ]; then
                echo "  Input directory missing: ${MISSING_INPUT_FILES[*]}"
            fi
            if [ ${#MISSING_OUTPUT_FILES[@]} -ne 0 ]; then
                echo "  Output directory missing: ${MISSING_OUTPUT_FILES[*]}"
            fi
        fi

        echo "----------------------------------------"
    fi
done
