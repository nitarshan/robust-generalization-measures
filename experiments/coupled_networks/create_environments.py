import numpy as np
import pickle

from collections import defaultdict
from math import isclose
from os import makedirs
from sys import argv
from tqdm import tqdm

from common import get_complexity_measures, hoeffding_weight, get_hps, load_data, sign_error


DATA_PATH = "../../data/nin.cifar10_svhn.csv"
SAVE_PATH = "environment_cache"
makedirs(SAVE_PATH, exist_ok=True)


def create_environments(datasets, testing_set_size, filter_noise=True):
    """
    Precomputation for environments.

    We compute the expectation over all random seeds of the sign error for every
    possible pair of hyperparameter combinations that differ in exactly one value.
    The resulting matrix is used in later code to produce the paper figures.

    """
    data = load_data(DATA_PATH)
    data = data.loc[[r["hp.dataset"] in datasets for _, r in data.iterrows()]]  # Select based on dataset

    # List of hyperparameters
    hps = get_hps(data)

    # Assign a unique index to each HP combination
    hp_combo_id = set([tuple(row[hps].values) for _, row in data.iterrows()])
    hp_combo_id = dict(zip(hp_combo_id, range(len(hp_combo_id))))

    # Get the name of every complexity measure (i.e., generalization measure) that should be considered
    c_measures = get_complexity_measures(data)

    # Extract all complexity measures and the generalization gap for each HP combo
    hp_c = [(tuple(row[hps].values), ({c: row[c] for c in c_measures}, row["gen.gap"])) for _, row in data.iterrows()]
    types_float_idx = np.where([isinstance(x, float) for x in list(hp_combo_id.keys())[0]])[0]  # Float-valued HP idx
    types_no_float_idx = np.where([not isinstance(x, float) for x in list(hp_combo_id.keys())[0]])[0]  # Other HP idx

    # Accumulators for the precomputation
    env_losses = {c: defaultdict(float) for c in c_measures}  # Weighted (unnormalized) sign error
    env_weights = defaultdict(float)  # Sum of weights
    env_weights_squared = defaultdict(float)  # Sum of squared weights

    # Calculate the sign error between every possible pair of distinct HP values
    # If there are multiple repeats for each combo, these will be averaged based
    # on their Hoeffding weight (for monte-carlo noise)
    for h1, (all_c1, g1) in tqdm(hp_c, position=0, leave=True):
        h1_id = hp_combo_id[h1]
        for h2, (all_c2, g2) in hp_c:
            h2_id = hp_combo_id[h2]

            # Dont compare pairs of identical HPs or those that differ by more than 1 value
            if (sum(not isclose(h1[i], h2[i]) for i in types_float_idx) +
                    sum(h1[i] != h2[i] for i in types_no_float_idx)) != 1:
                continue

            # Attribute a weight to the current pair of HPs based on the generalization gaps
            if filter_noise:
                # With Monte-Carlo noise filtering
                weight = hoeffding_weight(np.abs(g1 - g2), n=testing_set_size)

                # Discard all points with a weight below 0.5, i.e., (w - 0.5)_+
                min_weight = 0.5
                weight = weight - min_weight if weight > min_weight else 0
            else:
                # No Monte-Carlo noise filtering: all pairs get a weight of 1
                weight = 1

            # Accumulate weights needed to calculate effective sample size later
            env_weights[(h1_id, h2_id)] += weight
            env_weights_squared[(h1_id, h2_id)] += weight**2

            # Get the loss for each complexity measure
            for c in c_measures:
                c1 = all_c1[c]
                c2 = all_c2[c]
                env_losses[c][(h1_id, h2_id)] += sign_error(c1, c2, g1, g2) * weight

    # Save to disk
    pickle.dump({
        # A unique identifier for each hyperparameter combination
        "hp_combo_id": hp_combo_id,
        # List of hyperparameters in their order of appearance in the tuples that we save
        "hps": hps,
        # Sign-error (unnormalized) for each pair of hyperparameters
        "env_losses": env_losses,
        # The sum of weights for each pair of hyperparameters Later used to normalize and calculate the
        # effective sample size.
        "env_weights": {"raw": env_weights, "squared": env_weights_squared}
    }, open(SAVE_PATH + "/precomputations__filternoise%s__%s.pkl" % (filter_noise, "_".join(datasets)), "wb"))


if __name__ == "__main__":
    datasets = argv[1].split("_")
    available_datasets = ["cifar10", "svhn"]
    assert all(d in available_datasets for d in datasets)

    filter_noise = argv[2].lower() == "true"
    if not filter_noise:
        print("Warning: Monte Carlo noise filtering is disabled.")

    # Warning: we hardcode the test set size since it's the same for all our datasets. Adapt this to your setting.
    test_size = 10000

    create_environments(datasets=datasets, testing_set_size=test_size, filter_noise=filter_noise)
