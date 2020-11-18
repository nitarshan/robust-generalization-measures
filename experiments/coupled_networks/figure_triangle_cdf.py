import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from matplotlib.lines import Line2D
from sys import argv
from warnings import warn

from common import pretty_measure


ENVIRONMENT_CACHE_PATH = "./environment_cache"


pretty_hps = {"all": "All", "hp.lr": "lr", "hp.model_depth": "Depth", "hp.model_width": "Width",
              "hp.train_size": "TS", "hp.dataset": "Dataset"}


def triangle_cdf_plots_get_losses(hp, precomp, measure, min_ess=20.):
    """
    Get the value of the loss for each environment where a given hyperparameter is varied. The loss in each environment
    is a weighted expectation of sign-errors (as explained in the paper). The losses are grouped by pairs of values of
    the hyperparameter (v1 -> v2).

    Parameters:
    -----------
    hp: str
        The hyperparameter to vary
    precomp: dict
        The precomputations
    measure: str
        The generalization measure to evaluate
    min_ess: float
        The minimum effective sample size (default=12)

    Returns:
    --------
    box_losses: dict
        A dictionnary where keys are pairs of hp values and values are lists of all losses for
        environments where the hp varies from v1 -> v2.

    """
    if np.isclose(min_ess, 0):
        warn("Setting the minimum effective sample size to zero will lead to divisions by zero and is not recommended.")

    # Load the precomputations
    hp_idx = precomp["hps"].index(hp)
    hp_combo_id = precomp["hp_combo_id"]
    env_losses = precomp["env_losses"][measure]
    env_weights = precomp["env_weights"]["raw"]
    env_weights_squared = precomp["env_weights"]["squared"]

    # Unique values for the HP
    values = np.unique([combo[hp_idx] for combo in hp_combo_id.keys()])

    box_losses = {}

    for i, v1 in enumerate(values):
        # All points where Hi = v1
        v1_combos = [h for h in hp_combo_id if h[hp_idx] == v1]

        for j, v2 in enumerate(values):
            if v1 == v2:
                continue

            # Generate the coupled hp combos
            v2_combos = [v1c[: hp_idx] + (v2,) + v1c[hp_idx + 1:] for v1c in v1_combos]

            # Filter out v1_combos for which the v2_combo doesn't exist (e.g., job didn't finish)
            v1_combos_, v2_combos_, v1_idx, v2_idx = zip(*[(v1c, v2c, hp_combo_id[v1c], hp_combo_id[v2c]) for v1c, v2c
                                                           in zip(v1_combos, v2_combos) if v2c in hp_combo_id])

            # Get weights + sanity check
            sum_of_weights = np.array([env_weights[x] for x in zip(v1_idx, v2_idx)])
            sum_of_squared_weights = np.array([env_weights_squared[x] for x in zip(v1_idx, v2_idx)])
            sum_of_squared_weights[np.isclose(sum_of_weights, 0)] = 1  # Avoid division by zero and won't change result
            effective_sample_sizes = sum_of_weights**2 / sum_of_squared_weights

            # Filter out envs for which weight sum <= threshold
            losses = np.array([env_losses[x] for x in zip(v1_idx, v2_idx)])
            selector = np.logical_or(np.isclose(effective_sample_sizes, min_ess), effective_sample_sizes > min_ess)
            losses = losses[selector] / sum_of_weights[selector]  # Normalize weights to sum to 1 in the average

            box_losses[(v1, v2)] = losses

    return box_losses


def make_figure(datasets, measure, hp, min_ess=12, filter_noise=True):
    data_key = "_".join(datasets)

    precomp = pickle.load(open(ENVIRONMENT_CACHE_PATH + "/precomputations__filternoise%s__%s.pkl" %
                               (str(filter_noise).lower(), data_key), "rb"))

    box_losses = triangle_cdf_plots_get_losses(hp, precomp, measure, min_ess=min_ess)

    values = np.unique([vals[0] for vals in box_losses])

    f, axes = plt.subplots(ncols=len(values), nrows=len(values), sharex=True, sharey=True)
    cbar_ax = f.add_axes([.77, .127, .05, .55])
    for i, v1 in enumerate(values):
        for j, v2 in enumerate(values):
            if j >= i:
                axes[i, j].set_visible(False)
                continue

            bins = np.linspace(0, 1, 100)

            if len(box_losses[(v1, v2)]) != 0:
                # Calculate CDF
                z = np.zeros((len(bins), 1))
                for k, b in enumerate(bins):
                    z[k] = (box_losses[(v1, v2)] <= b).sum() / len(box_losses[(v1, v2)])

                heatmap = sns.heatmap(z, cmap="Blues_r", vmin=0.5, vmax=1, rasterized=True, ax=axes[i, j],
                                      cbar_ax=cbar_ax)

                axes[i, j].axhline(np.percentile(box_losses[(v1, v2)], q=90) * 100, color="magenta", linestyle="--",
                                   linewidth=1.5, zorder=2)
                axes[i, j].axhline(np.mean(box_losses[(v1, v2)]) * 100, color="orange", linestyle=":",
                                   linewidth=1.5, zorder=2)
                axes[i, j].axhline(np.max(box_losses[(v1, v2)]) * 100, color="limegreen", linewidth=1.5, zorder=1)
            else:
                # No data = no environment had a total weight ≥ min weight
                axes[i, j].scatter([0.5], [50], color="red", marker="x", s=30)

            axes[i, j].invert_yaxis()
            axes[i, j].set_ylim([-1, 102])
            axes[i, j].set_yticks([0, 50, 100])
            axes[i, j].set_yticklabels([0, 0.5, 1], fontsize=5, rotation=0)
            axes[i, j].set_xticklabels([], fontsize=6, rotation=0)
            axes[i, j].xaxis.set_visible(False)
            axes[i, j].yaxis.set_visible(False)

    # Set axis labels
    for i, v1 in enumerate(values):
        if hp != "hp.dataset":
            if isinstance(v1, float):
                axes[i, 0].set_ylabel("%s=%f" % (pretty_hps[hp], v1), fontsize=6)
                axes[-1, i].set_xlabel("%s=%f" % (pretty_hps[hp], v1), fontsize=6, rotation=45, ha="right")
            else:
                font_size = 5 if hp == "hp.model_width" else 6
                axes[i, 0].set_ylabel("%s=%d" % (pretty_hps[hp], v1), fontsize=font_size)
                axes[-1, i].set_xlabel("%s=%d" % (pretty_hps[hp], v1), fontsize=font_size, rotation=45, ha="right")
        else:
            axes[i, 0].set_ylabel("%s=%s" % (pretty_hps[hp], v1), fontsize=6)
            axes[-1, i].set_xlabel("%s=%s" % (pretty_hps[hp], v1), fontsize=6, rotation=45, ha="right")
        axes[i, 0].yaxis.set_visible(True)
        axes[-1, i].xaxis.set_visible(True)

    heatmap.collections[0].colorbar.ax.tick_params(labelsize=6)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    lines = [(Line2D([0], [0], color='limegreen', linewidth=1.5, linestyle='-'), 'max'),
             (Line2D([0], [0], color='magenta', linewidth=1.5, linestyle='--'), '90th percentile'),
             (Line2D([0], [0], color='orange', linewidth=1.5, linestyle=':'), 'mean')]
    plt.legend(*zip(*lines), loc='upper center', ncol=len(lines), bbox_to_anchor=(-6.7, 1.22), columnspacing=1,
               fontsize=4.5)

    f.set_size_inches(w=1.3, h=2.7)
    plt.savefig("figure_triangle_cdf__ds_%s__mess_%f__gm_%s__filternoise_%s_hp_%s.pdf" %
                (data_key, min_ess, pretty_measure(measure), str(filter_noise).lower(), hp), bbox_inches="tight")


if __name__ == "__main__":
    datasets = argv[1].split("_")
    available_datasets = ["cifar10", "svhn"]
    assert all(d in available_datasets for d in datasets)

    measure = argv[2]

    hp = argv[3]
    available_hps = ["hp.model_depth", "hp.model_width", "hp.lr", "hp.train_size", "hp.dataset"]
    if hp not in available_hps:
        raise ValueError("Invalid hyperparameter (hp) specified.")

    # Minimum effective sample size
    min_ess = float(argv[4])
    assert min_ess >= 0

    filter_noise = argv[5].lower() == "true"
    if not filter_noise:
        print("Warning: Monte Carlo noise filtering is disabled.")

    make_figure(datasets=datasets, measure=measure, hp=hp, min_ess=min_ess, filter_noise=filter_noise)
