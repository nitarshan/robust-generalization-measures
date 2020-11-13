import numpy as np
import pandas as pd


def average_over_repeats(data):
    """
    Take the expectation of generalization measures and errors over all points with the same hyperparameters

    """
    return data.groupby("experiment_id").mean()


def get_complexity_measures(data):
    """
    Get the name of all columns corresponding to complexity measures (i.e., generalization measures)

    """
    return [c for c in data.columns if c.startswith("complexity.")]


def get_hps(data):
    """
    Get the name of all columns corresponding to hyperparameters

    """
    return [c for c in data.columns if c.startswith("hp.")]


def hoeffding_weight(delta_gen, n=10000, shift=0):
    """
    This value has the following guarantee. If your measurement of the generalization gap is computed
    using n (say n=10,000) independent samples, then accepting samples only when this value > p would
    mean that those samples are legit different with probability at least p.

    Parameters:
    -----------
    delta_gen: float
        The absolute difference between two estimated generalization gaps
    n: int
        The size of the data sample used to estimate the generalization gaps

    Returns:
    --------
    weight: float
        Probability that the two generalization gaps are actually different

    """
    def phi(x, n):
        return 2 * np.exp(-2 * n * (x / 2)**2)
    return max(0., 1. - phi(max(np.abs(delta_gen) - shift, 0), n))**2


def load_data(data_path):
    """
    Load the data (outcome of training runs) for all models that meet crossentropy and accuracy standards.
    Discard those that do not.

    """
    def clean_data(data, name=''):
        # Discard measurements that do not meet crossentropy standards and warn
        # These might have reached the max number of epochs.
        n_before = data.shape[0]
        data = data.loc[data["is.converged"]]
        if data.shape[0] < n_before:
            print(f"[{name}] Warning: discarded %d results that did not meet the cross-entropy standards." %
                  (n_before - data.shape[0]))

        # Discard measurements that do not meet accuracy standards and warn
        n_before = data.shape[0]
        data = data.loc[data["is.high_train_accuracy"]]
        if data.shape[0] < n_before:
            print(f"[{name}] Warning: discarded %d results that did not meet the accuracy standards." %
                  (n_before - data.shape[0]))

        n_before = data.shape[0]
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if data.shape[0] < n_before:
            print(f"[{name}] Warning: discarded %d results that contained inf/nan values." % (n_before - data.shape[0]))

        return data

    # Load the data
    data = clean_data(pd.read_csv(data_path), "data")

    # Minor post-processing
    data["hp.train_size"] = data.train_dataset_size  # Rename column
    del data["hp.train_dataset_size"]
    data["hp.lr"] = data["hp.lr"].round(4)  # Needed in case some runs were computed accross difference devices

    return data


def sign_error(c1, c2, g1, g2):
    """
    This loss function measures the positive association between two data points ranked
    according to two scores c and g, i.e., a generalization measure and the generalization
    gap, respectively.

    Parameters:
    -----------
    c1, c2: float
        The value of the generalization measure for the two data points
    g1, g2: float
        The value of the generalization gap for the two data points

    Returns:
    --------
    loss: float
        This loss is bounded between 0 and 1. A value of 0 indicates that the points are
        ranked equally according to c and g. A positive value indicates that the rankings
        did not match.

    """
    error = float(np.sign(c1 - c2) * np.sign(g1 - g2))
    return (1 - error) / 2


def pretty_measure(c):
    """
    Pretty print measure names

    """
    if c == "complexity.params":
        c = "complexity.num.params"
    return c.replace("complexity.", "").replace("_", ".").replace("log.", "").replace(".fft", "")


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(0, 0.1, 1000)
    plt.plot(x, [hoeffding_weight(v, n=10000, shift=0) for v in x], label='$\\tau=0$')
    plt.plot(x, [hoeffding_weight(v, n=10000, shift=0.01) for v in x], label='$\\tau=1%$')
    plt.plot(x, [hoeffding_weight(v, n=10000, shift=0.02) for v in x], label='$\\tau=2%$')
    plt.legend()
    plt.show()
