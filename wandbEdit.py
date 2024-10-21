import wandb

# Login to wandb
wandb.login()

# Initialize the API
api = wandb.Api()

# Get all runs in the project
runs = api.runs("st7ma784/AllDataPGN")

# Update parameters for each run
# for run in runs:
#     run.config.update({"dataset": "tinyImageNet"})
#     run.update()

print("Updated all runs to use cifar10 dataset.")