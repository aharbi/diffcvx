import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


from model import EconomicDispatchModel, Generator
from forecast import Forecaster
from data import ForecastDataset
from metrics import compute_average_prediction

plt.style.use("figures/config.mplstyle")


def visualize_caiso_sample():
    """Plots a sample of the hourly electrical load of the CAISO network."""

    train_dir = "data/caiso_train.csv"
    test_dir = "data/caiso_test.csv"

    train_df = pd.read_csv(train_dir)
    test_df = pd.read_csv(test_dir)

    train_time_series = train_df["Load"]
    test_time_series = test_df["Load"]

    plt.figure()

    plt.plot(
        range(len(train_time_series)),
        train_time_series,
        linewidth=0.5,
        label="Training",
    )
    plt.plot(
        range(len(train_time_series), len(train_time_series) + len(test_time_series)),
        test_time_series,
        linewidth=0.5,
        label="Testing",
    )

    plt.xlabel("Time Index")
    plt.ylabel("Normalized Electrical Load")
    plt.legend()
    plt.savefig("figures/sample_caiso_load.pdf")


def visualize_ed_schedule():
    """Plots the hourly electrical generation schedule.

    Args:
        model (EconomicDispatchModel): economic dispatch model object.
    """

    power_system_specs = json.load(open("data/system.json"))
    generators = [Generator(specification) for specification in power_system_specs]

    model = EconomicDispatchModel(generators=generators, horizon=24)

    i = 400
    df = pd.read_csv("data/caiso_train.csv")
    load = df["Load"][i : (i + 24)].to_numpy()

    model.solve_ed(demand=load)

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


def visualize_loss():
    """Plots the loss function value over training iterations."""

    with open("models/loss_log_baseline.json", "r") as file:
        loss_baseline = json.load(file)

    with open("models/loss_log_end_to_end.json", "r") as file:
        loss_end_to_end = json.load(file)

    with open("models/loss_log_end_to_end_relaxed.json", "r") as file:
        loss_end_to_end_relaxed = json.load(file)

    plt.figure()

    plt.plot(loss_baseline, linewidth=0.5, label="Baseline")
    plt.plot(loss_end_to_end, linewidth=0.5, label="End-to-End")
    plt.plot(loss_end_to_end_relaxed, linewidth=0.5, label="End-to-End (Relaxed)")
    plt.xlabel("Iteration k")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig("figures/loss.pdf")


def visualize_predictions():
    """Plot the average prediction of the trained models along with standard deviation."""

    history_horizon = 24
    forecast_horizon = 24
    num_layers = 0
    num_hidden = 32

    testing_dir = "data/caiso_test.csv"
    column = "Load"

    testing_dataset = ForecastDataset(
        data_dir=testing_dir,
        column=column,
        history_horizon=history_horizon,
        forecast_horizon=forecast_horizon,
    )

    model_names = [
        "Baseline",
        "End-to-End",
        "End-to-End (CapEx)",
        "End-to-End (Reliability)",
    ]
    model_dirs = [
        "models/mlp_baseline_14.pth",
        "models/mlp_end_to_end_14.pth",
        "models/mlp_end_to_end_14_capex.pth",
        "models/mlp_end_to_end_14_ramping.pth",
    ]

    fig, ax = plt.subplots(2, 2, figsize=(6.75, 2.25))

    counter = 0

    for i in range(2):
        for j in range(2):

            forecaster = Forecaster(
                input_dim=history_horizon,
                output_dim=forecast_horizon,
                num_layers=num_layers,
                num_hidden=num_hidden,
                load_dir=model_dirs[counter],
            )

            mean, std, mean_true = compute_average_prediction(
                forecaster, testing_dataset
            )

            ax[i, j].plot(mean, label=model_names[counter])
            ax[i, j].plot(mean_true, "--", label="Ground Truth")

            ax[i, j].fill_between(
                range(24),
                mean - std,
                mean + std,
                alpha=0.3,
            )

            ax[i, j].set_xticks(range(0, 24, 2))
            ax[i, j].legend()

            counter += 1

        fig.supxlabel("Hour")
        fig.supylabel("Normalized Electrical Load")

    plt.savefig("figures/predictions.pdf")


if __name__ == "__main__":
    visualize_caiso_sample()
    visualize_ed_schedule()
    visualize_loss()
    visualize_predictions()
