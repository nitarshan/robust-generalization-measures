"""
An easier set of environments that involves more averaging (seed appendix)

"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from collections import defaultdict
from itertools import product
from matplotlib.lines import Line2D
from sys import argv

from common import load_data, pretty_measure


DATA_PATH = "../../data/nin.cifar10_svhn.csv"
ENVIRONMENT_CACHE_PATH = "./environment_cache"


def get_environment_losses(data, precomp, measure, min_ess):
    """
    Here, environments are the union of environments in the main text. That is, an environment varies one
    hyperparameter from v1 --> v2, but the remaining hyperparameters are allowed to vary between the pairs
    being compared. This is detailed in Appendix "Exploring a weaker family of environments".

    Parameters:
    -----------
    precomp: dict
        The precomputations
    measure: str
        The generalization measure to evaluate
    min_ess: float
        The minimum effective sample size (default=12)

    Returns:
    --------
    all_losses: array-like
        The losses for all environments

    """
    hps = precomp["hps"]
    hp_combo_id = precomp["hp_combo_id"]
    env_losses = precomp["env_losses"][measure]
    env_weights = precomp["env_weights"]["raw"]
    env_weights_squared = precomp["env_weights"]["squared"]

    combined_env_losses = defaultdict(float)
    combined_env_weights = defaultdict(float)
    combined_env_weights_squared = defaultdict(float)

    # Get i of Hi (in paper notation)
    for hp_varied in hps:

        # Get j of Hj (in paper notation)
        other_hps = list(set(hps).difference([hp_varied]))

        # Iterate over every possibly Hj
        for other_vals, other_data in data.groupby(other_hps):

            # All other hps are fixed, only hp_varied varies in other_data
            # We simply need to dispatch the data to the right environment
            for v1, v2 in product(np.unique(other_data[hp_varied]), np.unique(other_data[hp_varied])):
                # Never compare identical pairs
                if isinstance(v1, str) and isinstance(v2, str):
                    if v1 == v2:
                        continue
                else:
                    if np.isclose(v1, v2):
                        continue

                v1c = tuple(other_data.loc[other_data[hp_varied] == v1].iloc[0][hps].values)
                v2c = tuple(other_data.loc[other_data[hp_varied] == v2].iloc[0][hps].values)
                v1_idx = hp_combo_id[v1c]
                v2_idx = hp_combo_id[v2c]

                # XXX: We accumulate losses and weights from all environments to merge.
                #      The loss of each environment is a weighted sum, but the weights
                #      are not normalized yet. We divide by the total weight later to
                #      achieve this.
                combined_env_losses[(hp_varied, v1, v2)] += env_losses[(v1_idx, v2_idx)]
                combined_env_weights[(hp_varied, v1, v2)] += env_weights[(v1_idx, v2_idx)]
                combined_env_weights_squared[(hp_varied, v1, v2)] += env_weights_squared[(v1_idx, v2_idx)]

    # Renormalize weights in the combined environments
    for env in combined_env_losses.keys():
        if combined_env_weights[env] > 0:
            combined_env_losses[env] /= combined_env_weights[env]  # Normalize the weights in the expectation
        else:
            combined_env_losses[env] = np.inf

    # Sanity check to make sure we combined everything properly
    assert all((x >= 0 and x <= 1) for x in combined_env_losses.values())

    # Filter environments with too few significant pairs
    losses = np.array(list(combined_env_losses.values()))
    sum_of_weights = np.array([combined_env_weights[e] for e in combined_env_losses.keys()])
    sum_of_squared_weights = np.array([combined_env_weights_squared[e] for e in combined_env_losses.keys()])
    sum_of_squared_weights[np.isclose(sum_of_weights, 0)] = 1  # Avoid division by zero and won't change result
    effective_sample_sizes = sum_of_weights**2 / sum_of_squared_weights
    selector = np.logical_or(np.isclose(effective_sample_sizes, min_ess), effective_sample_sizes > min_ess)
    return losses[selector]


def make_figure(datasets, min_ess=12, filter_noise=True):
    data_key = "_".join(datasets)

    # Load the raw data (used to get losses in combined environments)
    data = load_data(DATA_PATH)
    data = data.loc[[r["hp.dataset"] in datasets for _, r in data.iterrows()]]  # Select based on dataset

    # Load the precomputations
    precomp = pickle.load(open(ENVIRONMENT_CACHE_PATH + "/precomputations__filternoise%s__%s.pkl" %
                               (str(filter_noise).lower(), data_key), "rb"))

    # Get the list of losses for each generalization measure
    # We will later report statistics of the distribution of losses for each measure
    losses = {}
    for c in list(precomp["env_losses"].keys()):
        print(c)

        # Use the FFT version of spectral measures if available
        if "_fft" not in c and c + "_fft" in precomp["env_losses"].keys():
            print("Skipping", c, "in favor of", c + "_fft")
            continue

        losses[c] = get_environment_losses(data, precomp=precomp, measure=c, min_ess=min_ess)

    # Order measures by mean sign error over all HPs
    ordered_measures = np.array(list(losses.keys()))[np.argsort([np.max(losses[c]) for c in losses.keys()])].tolist()

    # Make plot
    f = plt.gcf()
    ax = plt.gca()

    bins = np.linspace(0, 1, 100)
    cbar_ax = f.add_axes([.91, .127, .02, .75])

    z = np.zeros((len(bins), len(ordered_measures)))
    for i, c in enumerate(ordered_measures):
        # Get losses
        l = losses[c]

        # Plot mean and max
        ax.axvline(i, linestyle="-", color="white", linewidth=3, zorder=999)

        if len(l) > 0:
            ax.plot([i, i + 1], [np.mean(l) * 100, np.mean(l) * 100], color="orange", zorder=2, linewidth=1.5,
                    linestyle=":")
            ax.plot([i, i + 1], [np.percentile(l, q=90) * 100, np.percentile(l, q=90) * 100], color="magenta",
                    zorder=2, linewidth=1.5, linestyle="--")
            ax.plot([i, i + 1], [np.max(l) * 100, np.max(l) * 100], color="limegreen", zorder=1, linewidth=1.5)

            # Calculate CDF
            for j, b in enumerate(bins):
                z[j, i] = (l <= b).sum() / len(l)
        else:
            # No data = no environment had a total weight ≥ min weight
            ax.scatter([i + 0.5], [50], marker="x", color="red")

        if z.sum() > 0:
            heatmap = sns.heatmap(z, cmap="Blues_r", vmin=0.5, vmax=1, rasterized=True, ax=ax, cbar_ax=cbar_ax)
            heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)

        ax.invert_yaxis()
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels([0, 0.5, 1], fontsize=6, rotation=0)
        ax.set_ylabel("Sign-error distribution\n(%d)" % len(list(losses.values())[0]), fontsize=8)
        ax.set_ylim(0, 101)

    ax.set_xticks(np.arange(len(ordered_measures)) + 0.5)
    ax.set_xticklabels([pretty_measure(c) for c in ordered_measures], rotation=45, fontsize=8, ha="right")

    lines = [(Line2D([0], [0], color='limegreen', linewidth=1.5, linestyle='-'), 'max'),
             (Line2D([0], [0], color='magenta', linewidth=1.5, linestyle='--'), '90th percentile'),
             (Line2D([0], [0], color='orange', linewidth=1.5, linestyle=':'), 'mean')]
    plt.legend(*zip(*lines), loc='upper center', ncol=len(lines), bbox_to_anchor=(-19.5, 1.1), labelspacing=10,
               fontsize=8)

    f.set_size_inches(w=10, h=4.8)
    plt.savefig("figure__signerror_cdf_per_hp_easy_envs__ds_%s__mess_%f__filternoise_%s_cdf_per_hp.pdf" %
                (data_key, min_ess, str(filter_noise).lower()), bbox_inches="tight")


if __name__ == "__main__":
    datasets = argv[1].split("_")
    available_datasets = ["cifar10", "svhn"]
    assert all(d in available_datasets for d in datasets)

    # Minimum effective sample size
    min_ess = float(argv[2])
    assert min_ess >= 0

    filter_noise = argv[3].lower() == "true"
    if not filter_noise:
        print("Warning: Monte Carlo noise filtering is disabled.")

    make_figure(datasets=datasets, min_ess=min_ess, filter_noise=filter_noise)
