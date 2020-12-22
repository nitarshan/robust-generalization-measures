# %% Imports
import argparse
from pathlib import Path

import wandb

# %% Flags
parser = argparse.ArgumentParser(description='Export Wandb results')
parser.add_argument('--tag', type=str)
parser.add_argument('--entity', type=str)
flags = parser.parse_args()
tag = flags.tag
entity = flags.entity

# %% Wandb connection
# https://docs.wandb.com/library/api/examples#export-metrics-from-all-runs-in-a-project-to-a-csv-file
api = wandb.Api()
runs = api.runs(f"{entity}/rgm_single", {"config.wandb_tag": tag})
print(f"Found {len(runs)} runs for tag {tag}")

summary_list = [] 
config_list = [] 
name_list = [] 
for run in runs: 
    # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict) 

    # run.config is the input metrics.  We remove special values that start with _.
    config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')}) 

    # run.name is the name of the run.
    name_list.append(run.name)       

# %% Save CSV files
import pandas as pd 
summary_df = pd.DataFrame.from_records(summary_list) 
config_df = pd.DataFrame.from_records(config_list) 
name_df = pd.DataFrame({'name': name_list}) 
all_df = pd.concat([name_df, config_df,summary_df], axis=1)

resultspath = Path(f'temp/single_network/')
resultspath.mkdir(parents=True, exist_ok=True)

all_df.to_csv(resultspath / f"{tag}_export.csv")
