import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from matplotlib.lines import Line2D
from sys import argv
from warnings import warn

from common import pretty_measure


ENVIRONMENT_CACHE_PATH = "./environment_cache"


def get_all_losses(hp, precomp, measure, min_ess=12.):
    """
    Get the value of the loss for each environment where a given hyperparameter is varied. The loss in each environment
    is a weighted expectation of sign-errors (as explained in the paper).

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
    all_losses: array-like
        The list of all losses where the hp is varied

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

    all_losses = []

    for i, v1 in enumerate(values):
        # All points where Hi = v1
        v1_combos = [h for h in hp_combo_id if h[hp_idx] == v1]

        for j, v2 in enumerate(values):
            if v1 == v2 or (isinstance(v1, float) and np.isclose(v1, v2)):
                continue

            # Generate the coupled hp combos
            # Hi: v1 -> v2 and the rest (Hj) is constant
            v2_combos = [v1c[: hp_idx] + (v2,) + v1c[hp_idx + 1:] for v1c in v1_combos]

            # Filter out v1_combos for which the v2_combo doesn't exist (e.g., job didn't finish)
            v1_combos_, v2_combos_, v1_idx, v2_idx = zip(*[(v1c, v2c, hp_combo_id[v1c], hp_combo_id[v2c]) for v1c, v2c
                                                           in zip(v1_combos, v2_combos) if v2c in hp_combo_id])

            # Note: Each pair of v1 and v2 HP combos forms an environment where Hi: v1 -> v2 and Hj is constant.
            #       Averaging over repeats is assume to be done a priori in the precomputation.

            # Get weights + sanity check
            sum_of_weights = np.array([env_weights[x] for x in zip(v1_idx, v2_idx)])
            sum_of_squared_weights = np.array([env_weights_squared[x] for x in zip(v1_idx, v2_idx)])
            sum_of_squared_weights[np.isclose(sum_of_weights, 0)] = 1  # Avoid division by zero and won't change result
            effective_sample_sizes = sum_of_weights**2 / sum_of_squared_weights
            assert np.isclose(np.abs(env_weights[(
                v1_idx[0], v2_idx[0])] - env_weights[(v2_idx[0], v1_idx[0])]).sum(), 0)

            # Get losses + sanity check
            losses = np.array([env_losses[x] for x in zip(v1_idx, v2_idx)])
            assert np.isclose(np.abs(env_losses[(v1_idx[0], v2_idx[0])] - env_losses[(v2_idx[0], v1_idx[0])]).sum(), 0)

            # Filter out envs for which weight sum <= threshold
            selector = np.logical_or(np.isclose(effective_sample_sizes, min_ess), effective_sample_sizes > min_ess)
            losses = losses[selector] / sum_of_weights[selector]  # Normalize weights to sum to 1 in the average

            # Sanity check
            if len(losses) > 0:
                assert np.nanmax(losses) <= 1
                assert np.nanmin(losses) >= 0
            all_losses += losses.tolist()

    # Sanity check since we should be seeing each pair (v1, v2) twice
    assert len(all_losses) % 2 == 0

    return np.array(all_losses)


def make_figure(datasets, min_ess=12, filter_noise=True):
    data_key = "_".join(datasets)

    # Load precomputations
    precomp = pickle.load(open(ENVIRONMENT_CACHE_PATH + "/precomputations__filternoise%s__%s.pkl" %
                               (str(filter_noise).lower(), data_key), "rb"))

    # Get the losses for each generalization measure (also called complexity measure here), per hp
    complexity_losses_per_hp = {}
    for c in precomp["env_losses"].keys():

        # Use the FFT version of spectral measures if available
        if "_fft" not in c and c + "_fft" in precomp["env_losses"].keys():
            print("Skipping", c, "in favor of", c + "_fft")
            continue

        # For the current generalization measure, get the sign-errors in each environment where an
        # HP varies and store this per HP
        complexity_losses_per_hp[c] = {}
        for hp in precomp["hps"]:
            complexity_losses_per_hp[c][hp] = get_all_losses(hp, precomp, measure=c, min_ess=min_ess)
        complexity_losses_per_hp[c]["all"] = np.hstack([complexity_losses_per_hp[c][h] for h in
                                                        complexity_losses_per_hp[c]])

        # Sanity check
        assert complexity_losses_per_hp[c]["all"].shape[0] == \
            sum(complexity_losses_per_hp[c][hp].shape[0] for hp in precomp["hps"])

    # Order measures by mean sign error over all HPs
    ordered_measures = \
        np.array(list(complexity_losses_per_hp.keys()))[np.argsort([np.mean(complexity_losses_per_hp[c]["all"])
                                                                    for c in complexity_losses_per_hp])].tolist()
    # ordered_measures = np.sort(list(env_losses.keys())).tolist()  # Uncomment to order by name

    # Ordering and rendering used in the plot
    ordered_hps = ["all", "hp.lr", "hp.model_depth", "hp.model_width", "hp.train_size", "hp.dataset"]
    pretty_hps = {"all": "All", "hp.lr": "LR", "hp.model_depth": "Depth", "hp.model_width": "Width",
                  "hp.train_size": "Train size", "hp.dataset": "Dataset"}

    # Don't plot dataset axis if there is only a single one
    if len(datasets) == 1:
        ordered_hps.remove("hp.dataset")
        precomp["hps"].remove("hp.dataset")

    bins = np.linspace(0, 1, 100)
    f, axes = plt.subplots(ncols=1, nrows=len(ordered_hps), sharex=True, sharey=True)
    cbar_ax = f.add_axes([.91, .127, .02, .75])
    for ax, hp in zip(axes, ordered_hps):
        z = np.zeros((len(bins), len(ordered_measures)))
        for i, c in enumerate(ordered_measures):
            # Get losses
            losses = complexity_losses_per_hp[c][hp]

            # Plot mean and max
            ax.axvline(i, linestyle="-", color="white", linewidth=3, zorder=999)

            if len(losses) > 0:
                # We need to multiply by 100 for the lines to appear in the correct place, because the heatmap's y-axis
                # goes from 0 to 100 (number of bins) and loss goes from 0 to 1.
                ax.plot([i, i + 1], [np.mean(losses) * 100, np.mean(losses) * 100], color="orange", zorder=2,
                        linewidth=1.5, linestyle=":")
                ax.plot([i, i + 1], [np.percentile(losses, q=90) * 100, np.percentile(losses, q=90) * 100],
                        color="magenta", zorder=2, linewidth=1.5, linestyle="--")
                ax.plot([i, i + 1], [np.max(losses) * 100, np.max(losses) * 100], color="limegreen", zorder=1,
                        linewidth=1.5)

                # Calculate CDF
                for j, b in enumerate(bins):
                    z[j, i] = (losses <= b).sum() / len(losses)
            else:
                # No data = no environment had a total weight ≥ min weight
                ax.scatter([i + 0.5], [50], marker="x", color="red")

        if z.sum() > 0:
            heatmap = sns.heatmap(z, cmap="Blues_r", vmin=0.5, vmax=1, rasterized=True, ax=ax, cbar_ax=cbar_ax)
            heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)

        ax.invert_yaxis()  # Seaborn will by default range the y axis from 100 to 0
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels([0, 0.5, 1], fontsize=6, rotation=0)
        ax.set_ylabel(pretty_hps[hp] + "\n(%d)" %
                      (len(complexity_losses_per_hp[list(complexity_losses_per_hp.keys())[0]][hp])), fontsize=8)
        ax.set_ylim(-1, 102)

    axes[-1].set_xticks(np.arange(len(ordered_measures)) + 0.5)
    axes[-1].set_xticklabels([pretty_measure(c) for c in ordered_measures], rotation=45, fontsize=8, ha="right")

    lines = [(Line2D([0], [0], color='limegreen', linewidth=1.5, linestyle='-'), 'max'),
             (Line2D([0], [0], color='magenta', linewidth=1.5, linestyle='--'), '90th percentile'),
             (Line2D([0], [0], color='orange', linewidth=1.5, linestyle=':'), 'mean')]
    plt.legend(*zip(*lines), loc='upper center', ncol=len(lines), bbox_to_anchor=(-19.5, 1.1), labelspacing=10,
               fontsize=8)

    f.set_size_inches(w=10, h=4.8)
    plt.savefig("figure__signerror_cdf_per_hp__ds_%s__mess_%f__filternoise_%s_cdf_per_hp.pdf" %
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
