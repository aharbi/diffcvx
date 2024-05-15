import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import EconomicDispatchModel

plt.rcParams["text.usetex"] = True


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

    plt.figure(figsize=(7, 3), dpi=200, layout="constrained")

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

    plt.title("Hourly CAISO Network Load (January 2021 - December 2023)")
    plt.xlabel("Time Index")
    plt.ylabel("Electrical Load (MWh)")
    plt.legend()
    plt.savefig("figures/sample_caiso_load.png")


def visualize_ed_schedule(model: EconomicDispatchModel):
    """Plots the hourly electrical generation schedule.

    Args:
        model (EconomicDispatchModel): economic dispatch model object.
    """

    generation = model.g.value
    demand = model.d.value

    m, n = generation.shape

    labels = [f"Generator {index + 1}" for index in range(n)]

    plt.figure(figsize=(7, 3), dpi=200, layout="constrained")

    plt.stackplot(range(n), generation, labels=labels)
    plt.plot(range(n), demand, linestyle="--", c="k", label="Demand")

    plt.title("Hourly Generator Schedule")
    plt.xlabel("Time (Hour)")
    plt.ylabel("Electricity Generation (MWh)")
    plt.legend(ncol=2)
    plt.savefig("figures/sample_ed_schedule.png")
