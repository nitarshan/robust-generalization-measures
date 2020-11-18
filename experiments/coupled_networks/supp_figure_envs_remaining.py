"""
Supplementary figure that shows the number of environments remaining when varying the
minimum effective sample size parameter

"""
import matplotlib.pyplot as plt
import numpy as np
import pickle

from collections import defaultdict
from sys import argv

# Use the same function as in the big CDF plot to get loss per environment
from figure_cdf_all_measures import get_all_losses


ENV_CACHE = "./environment_cache/"


# Command line arguments
datasets = argv[1].split("_")
lower = int(argv[2])  # Smallest effective sample size (ESS) cutoff to plot
upper = int(argv[3])  # Same as above, but largest ESS
assert lower < upper

pretty_hps = {"all": "All", "hp.lr": "Learning rate", "hp.model_depth": "Depth", "hp.model_width": "Width",
              "hp.train_size": "Train size", "hp.dataset": "Dataset"}

effective_sample_sizes = range(lower, upper + 1)

n_envs_per_hp = defaultdict(list)
for ess in effective_sample_sizes:
    precomp = pickle.load(open(ENV_CACHE + "/precomputations__filternoisetrue__%s.pkl" % "_".join(datasets), "rb"))

    # Get the losses for each complexity measure, per hp
    for hp in precomp["hps"]:
        losses = get_all_losses(hp, precomp,
                                measure="complexity.pacbayes_orig",  # the measure doesn't matter here, any is fine
                                min_ess=ess)
        n_envs_per_hp[hp].append(len(losses))

plt.clf()
for hp in n_envs_per_hp:
    plt.plot(np.array(effective_sample_sizes), n_envs_per_hp[hp], label=pretty_hps[hp])
plt.axvline(12, color="red", linestyle="--")

plt.xlabel(r"Minimum effective sample size")
plt.ylabel("Remaining environments")
plt.legend()
plt.gcf().set_size_inches(w=5, h=3)
plt.savefig("figure_miness_remaining_environments_ds_%s_range_%d_%d.pdf" % ("_".join(datasets), lower, upper),
            bbox_inches="tight")
