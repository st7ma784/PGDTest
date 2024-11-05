DIRECTORY="./slurm_scripts"


for folder in "$DIRECTORY"/*; do
    if [ -d "$folder" ]; then  # Check if it's a directory
        for file in "$folder"/*; do
            if [ -f "$file" ]; then  # Check if it's a file
                echo "Submitting job $file"
                #if hostname ends in bede.dur.ac.uk
                if [[ $(hostname) == *".bede.dur.ac.uk" ]]; then
                    sed -i '/module add opence/d' "$file"
                    echo "Running on bede"
                sbatch "$file"
                echo "Job $file submitted"
            fi
        done
    fi
done