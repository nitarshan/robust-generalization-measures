# %% Imports
import argparse
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Flags
parser = argparse.ArgumentParser(description='Export Wandb results')
parser.add_argument('--tag', type=str)
flags = parser.parse_args()
tag = flags.tag

# %% Helper functions
def get_model_results(results):
    env_split_mode = {}
    for split, split_results in results.groupby("env_split"):
        env_split_mode[split] = split_results[["actual_measure", "risk_max"]]
    return env_split_mode

def get_best_by_measure(data):
    data = pd.DataFrame(data).reset_index()
    measures = [c for c in data.actual_measure.unique() if len(c.split(".")) == 2]
    data["measure"] = data.actual_measure.apply(lambda x: ".".join(x.split(".")[:2]))
    data = data.iloc[data.groupby("measure").risk_max.idxmin()]
    return data

def preprocess_columns(data):
    data["mae_max"] = np.sqrt(data.risk_max)
    data["pretty_measure"] = [c.replace("complexity.", "").replace("_adjusted1", "").replace("_", "-") for c in data.actual_measure]
    return data

def subtract_baseline(data, baseline_mae):
    data["mae_max_vs_baseline"] = baseline_mae - data["mae_max"]
    return data

# %% Load data
print(tag)

resultspath = Path(f'temp/single_network/')
resultspath.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(resultspath / f'{tag}_export.csv')[['lr', 'bias', 'datafile', 'env_split', 'actual_measure', 'only_bias__ignore_input', 'selected_single_measure', 'bias.1', 'loss', 'weight', '_runtime', 'risk_max', 'risk_min', 'train_mse', 'risk_range', 'robustness_penalty']]
affine = get_model_results(df[(df['bias']==True) & (df['only_bias__ignore_input']==False)].copy())
weight_only = get_model_results(df[(df['bias']==False) & (df['only_bias__ignore_input']==False)].copy())
bias_only = get_model_results(df[(df['bias']==True) & (df['only_bias__ignore_input']==True)].copy())

# %% Plot regression results
sns.set_style("darkgrid", {'xtick.bottom': True})
plotpath = Path(f'temp/single_network/{tag}/')
plotpath.mkdir(parents=True, exist_ok=True)
order = None

for idx, split in enumerate(sorted(df.env_split.unique())):
    plt.figure(figsize=(8,1.5))
    baselines_bias = {split: preprocess_columns(bias_only[split]).mae_max.values[0] for split in bias_only}
    plot_results = preprocess_columns(get_best_by_measure(affine[split]))
    plot_results = subtract_baseline(plot_results, baseline_mae=baselines_bias[split])
    order = plot_results.sort_values("mae_max_vs_baseline", ascending=False).pretty_measure if order is None else order
    sns.barplot(data=plot_results, order=order, x="pretty_measure", y="mae_max", palette="deep")
    plt.axhline(baselines_bias[split], label='bias-only baseline')
    plt.xticks(rotation=90)
    #plt.title(f'split {split}')
    plt.legend(loc='lower right')
    plt.xlabel('Generalization Measure')
    plt.ylabel('Robust RMSE')
    plt.tight_layout()
    plt.xticks(rotation=45,ha='right')
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.savefig(plotpath / f'{split}_mae_all_vs_baseline.pdf', bbox_inches='tight')
    plt.close()

# %% Plot regression cdfs
D = namedtuple('D', ['measure', 'env_split', 'exp_type', 'bias_only'])

sns.set()
exp_type = tag.split('_')[-1]
rows = 1 if exp_type=='v1' else 5
plt.figure(figsize=(10,2 * rows))
sorting = None
for row, env_split in enumerate(['all', 'lr', 'depth', 'width', 'train_size']):
    if exp_type=='v1' and env_split != 'all':
        continue
    data = [(D(*x.name.split('__')), np.sqrt(np.load(x))) for x in Path('temp/single_network/risks').glob(f'*__{env_split}__{exp_type}__False.npy')]
    baseline = [(D(*x.name.split('__')), np.sqrt(np.load(x))) for x in Path('temp/single_network/risks').glob(f'*__{env_split}__{exp_type}__True.npy')]
    data[0][0], len(data)
    if sorting is None:
        data = sorted(data, key=lambda x: x[1].max())
        sorting = [x[0] for x in data]
    else:
        data_dict = dict(data)
        data = [(x._replace(env_split=env_split), data_dict[x._replace(env_split=env_split)]) for x in sorting]

    maxx = baseline[0][1].max()
    points = 100
    for i in range(len(data)):
        maxx = max(maxx, data[i][1].max())
    for i in range(len(data)):
        plt.subplot(rows,24,24*row + i+1)
        x = np.cumsum(np.histogram(data[i][1], points, (0,maxx))[0])
        x = x / x[-1]
        ax = sns.heatmap(x[..., np.newaxis], cmap="Blues_r", cbar=(i+1)==len(data), cbar_kws={"aspect":35}, rasterized=True)
        ax.invert_yaxis()
        plt.axhline(baseline[0][1].max()*points/maxx, color='red', label='baseline')
        plt.axhline(np.max(data[i][1])*points/maxx, color="limegreen", zorder=1, linewidth=1.5, label='max')
        plt.axhline(np.percentile(data[i][1], q=90)*points/maxx, color="magenta", zorder=2, linewidth=1.5, linestyle="--", label='90th percentile')
        plt.axhline(np.mean(data[i][1])*points/maxx, color='orange', zorder=2, linewidth=1.5, linestyle=":", label='mean')
        plt.ylabel('')
        if i==0:
            plt.ylabel(f"RMSE ({env_split.replace('_', ' ')})")
            plt.yticks([0, points//2, points], labels=[0, str(maxx/2)[:5], str(maxx)[:5]], fontsize=8)
        else:
            plt.yticks([])
        plt.xticks([])
        if row+1 == rows:
            plt.xlabel(data[i][0].measure.replace('_','.'), rotation=45, fontsize=8, ha="right")
        else:
            plt.xlabel('')
plt.legend(loc='upper left', ncol=4, bbox_to_anchor=(-25.5,-0.9), fontsize=8)
plt.savefig(plotpath / f'cdf_{exp_type}.pdf', bbox_inches='tight')
plt.close()
