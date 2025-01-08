#!/bin/bash

# List of scripts to run
scripts=("src/process_data.py" "src/prepare_features.py" "src/train_models.py" "src/infer.py")

# Loop through each script
for script in "${scripts[@]}"; do
    echo "Starting $script..."
    
    if python3 "$script"; then
        echo "$script completed successfully"
    else
        echo "Error running $script"
        exit 1  # Exit if any script fails
    fi
    
    echo "-------------------"
done

echo "All scripts completed"