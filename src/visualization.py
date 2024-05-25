import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import EconomicDispatchModel

plt.style.use("figures/config.mplstyle")


def visualize_caiso_sample(train_dir: str, test_dir: str):
    """Plots a sample of the hourly electrical load of the CAISO network.

    Args:
        train_dir (str): path to the training data csv file.
        test_dir (str): path to the testing data csv file.
    """

    train_df = pd.read_csv(train_dir)
    test_df = pd.read_csv(test_dir)

    train_time_series = train_df["Load"]
    test_time_series = test_df["Load"]

    plt.figure()

    plt.plot(
        range(len(train_time_series)),
        train_time_series,
        c="dodgerblue",
        linewidth=0.5,
        label="Training",
    )
    plt.plot(
        range(len(train_time_series), len(train_time_series) + len(test_time_series)),
        test_time_series,
        c="crimson",
        linewidth=0.5,
        label="Testing",
    )

    plt.xlabel("Time Index")
    plt.ylabel("Normalized Electrical Load")
    plt.legend()
    plt.xticks([1000, 3000, 6000, 8000, 10000])
    plt.savefig("figures/sample_caiso_load.pdf")


def visualize_ed_schedule(model: EconomicDispatchModel):
    """Plots the hourly electrical generation schedule.

    Args:
        model (EconomicDispatchModel): economic dispatch model object.
    """

    generation = model.g.value
    demand = model.d.value

    m, n = generation.shape

    plt.figure()

    plt.stackplot(range(n), generation)
    plt.plot(range(n), demand, linestyle="--", c="k", label="Demand")

    plt.xlabel("Time (Hour)")
    plt.ylabel("Normalized Generation")
    plt.legend()
    plt.savefig("figures/sample_ed_schedule.pdf")
