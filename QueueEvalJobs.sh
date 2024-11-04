DIRECTORY="./slurm_scripts"


for folder in "$DIRECTORY"/*; do
    if [ -d "$folder" ]; then  # Check if it's a directory
        for file in "$folder"/*; do
            if [ -f "$file" ]; then  # Check if it's a file
                echo "Submitting job $file"
                sbatch "$file"
                echo "Job $file submitted"
            fi
        done
    fi
done