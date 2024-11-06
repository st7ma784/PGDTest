#!/bin/bash
FOLDER=$1 # Folder to move is the first argument

# Loop through each folder in the current directory
for dir in $FOLDER/*; do
    # Find all files in the directory
    find "$dir" -type f | while read -r file; do
        # Replace ./results with $global_scratch/results
        sed -i "s|./results|$global_scratch/results|g" "$file"
        # Replace ./MODELDIR with $global_scratch/MODELDIR
        sed -i "s|./MODELDIR|$global_scratch/MODELDIR|g" "$file"
    done
done