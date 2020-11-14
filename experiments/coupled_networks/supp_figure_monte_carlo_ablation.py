import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from sys import argv

from common import pretty_measure
from figure_cdf_all_measures import get_all_losses


ENVIRONMENT_CACHE_PATH = "./environment_cache"


def get_complexity_losses_per_hp(datasets, min_ess, filter_noise):
    """
    Get the loss of each environment for each generalization measure grouped based on the
    HP that varies in each environment.

    """
    data_key = "_".join(datasets)

    precomp = pickle.load(open(ENVIRONMENT_CACHE_PATH + "/precomputations__filternoise%s__%s.pkl" %
                               (filter_noise, data_key), "rb"))

    # Get the losses for each complexity measure, per hp
    complexity_losses_per_hp = {}
    for c in precomp["env_losses"].keys():

        # Use the FFT version of spectral measures if available
        if "_fft" not in c and c + "_fft" in precomp["env_losses"].keys():
            print("Skipping", c, "in favor of", c + "_fft")
            continue

        complexity_losses_per_hp[c] = {}
        for hp in precomp["hps"]:
            complexity_losses_per_hp[c][hp] = get_all_losses(hp, precomp, measure=c, min_ess=min_ess)

        complexity_losses_per_hp[c]["all"] = np.hstack([complexity_losses_per_hp[c][h] for h in
                                                        complexity_losses_per_hp[c]])
        # Sanity check
        assert complexity_losses_per_hp[c]["all"].shape[0] == \
            sum(complexity_losses_per_hp[c][hp].shape[0] for hp in precomp["hps"])

    return complexity_losses_per_hp


def make_figure(datasets, min_ess=12, aggregate=np.mean, aggregate_name="Mean", color="blue", no_xlabel=False):
    losses_per_hp_with_noise = get_complexity_losses_per_hp(datasets, min_ess, filter_noise=False)
    losses_per_hp_without_noise = get_complexity_losses_per_hp(datasets, min_ess, filter_noise=True)

    # Ordered by name
    ordered_measures = np.sort(list(losses_per_hp_with_noise.keys())).tolist()

    # Collect plot data
    d = [aggregate(losses_per_hp_without_noise[c]['all']) - aggregate(losses_per_hp_with_noise[c]['all'])
         for c in ordered_measures]

    # Make plot
    plt.clf()
    sns.barplot(x=ordered_measures, y=d, color=color)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.ylim(-np.max(np.abs(d)), np.max(np.abs(d)))
    plt.ylabel("%s(w/ filter) - %s(w/o filter)" % (aggregate_name, aggregate_name), fontsize=10)
    if not no_xlabel:
        plt.gca().set_xticklabels([pretty_measure(c) for c in ordered_measures], rotation=45, fontsize=10, ha="right")
    else:
        plt.gca().set_xticklabels([""] * len(ordered_measures))
    plt.gcf().set_size_inches(w=8, h=2.5)
    plt.savefig("figure_monte_carlo_noise_ablation__%s__ds_%s__mess_%f_cdf.pdf" %
                (aggregate_name.lower().replace("$", "").replace("_", "").replace("{", "").replace("}", ""),
                 "_".join(datasets), min_ess), bbox_inches="tight")


if __name__ == "__main__":
    datasets = argv[1].split("_")
    available_datasets = ["cifar10", "svhn"]
    assert all(d in available_datasets for d in datasets)

    min_ess = float(argv[2])
    assert min_ess >= 0

    make_figure(datasets=datasets, min_ess=min_ess, aggregate=np.mean, aggregate_name="mean",
                color="blue", no_xlabel=True)
    make_figure(datasets=datasets, min_ess=min_ess, aggregate=np.median, aggregate_name=r"$P_{50}$",
                color="red", no_xlabel=True)
    make_figure(datasets=datasets, min_ess=min_ess, aggregate=lambda x: np.percentile(x, q=90),
                aggregate_name=r"$P_{90}$", color="orange", no_xlabel=True)
    make_figure(datasets=datasets, min_ess=min_ess, aggregate=np.max, aggregate_name="max",
                color="green", no_xlabel=False)
