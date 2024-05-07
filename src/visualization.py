import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


plt.rcParams["text.usetex"] = True


def plot_caiso_sample(train_dir: str, test_dir: str):
    """Plots a sample of the hourly electrical load of the CAISO network.

    Args:
        train_dir (str): path to the training data csv file.
        test_dir (str): path to the testing data csv file.
    """

    train_df = pd.read_csv(train_dir)
    test_df = pd.read_csv(test_dir)

    train_time_series = train_df["Load"]
    test_time_series = test_df["Load"]

    plt.figure(figsize=(7, 3), dpi=200)

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

    plt.title("Hourly CAISO Network Load (2022-2023)")
    plt.ylabel("Electrical Load (MWh)")
    plt.xlabel("Time Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/sample_caiso_load.png")


if __name__ == "__main__":

    train_dir = "data/caiso_train.csv"
    test_dir = "data/caiso_test.csv"

    plot_caiso_sample(train_dir, test_dir)
