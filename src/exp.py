import pandas as pd
import json

from forecast import *
from model import *
from visualization import *
from train import *
from metrics import *
from data import get_caiso


def get_data():
    # Training data
    train_start_date = pd.Timestamp("January 1, 2021").normalize()
    train_end_date = pd.Timestamp("June 30, 2023").normalize()
    train_save_dir = "data/caiso_train.csv"

    _ = get_caiso(train_start_date, train_end_date, train_save_dir)

    # Testing data
    test_start_date = pd.Timestamp("July 1, 2023").normalize()
    test_end_date = pd.Timestamp("December 31, 2023").normalize()
    test_save_dir = "data/caiso_test.csv"

    _ = get_caiso(test_start_date, test_end_date, test_save_dir)


def visualize_data():
    # Plot a sample of the CAISO data
    train_dir = "data/caiso_train.csv"
    test_dir = "data/caiso_test.csv"

    visualize_caiso_sample(train_dir, test_dir)

    # Plot a sample electricity generation schedule
    power_system_specs = json.load(open("data/system.json"))
    generators = [Generator(specification) for specification in power_system_specs]

    model = EconomicDispatchModel(generators=generators, horizon=24)

    i = 100
    df = pd.read_csv("data/caiso_train.csv")
    load = (df["Load"][i : (i + 24)] * 0.08).to_numpy()

    model.solve_ed(demand=load)

    visualize_ed_schedule(model)


def independent_model(train=False):
    # Parameters
    history_horizon = 72
    forecast_horizon = 24

    num_layers = 4
    num_hidden = 256

    num_epochs = 10
    batch_size = 128
    lr = 1e-4
    device = "cpu"

    lambdas = [0, 0]

    training_dir = "data/caiso_train.csv"
    testing_dir = "data/caiso_test.csv"

    system_dir = "data/system.json"

    column = "Load"
    save_dir = "models/"
    load_dir = "models/mlp_49.pth"

    # ED model
    power_system_specs = json.load(open(system_dir))
    generators = [Generator(specification) for specification in power_system_specs]

    ed_model = EconomicDispatchModel(
        generators=generators, horizon=forecast_horizon, lambdas=lambdas
    )

    # Training model
    training_dataset = ForecastDataset(
        data_dir=training_dir,
        column=column,
        history_horizon=history_horizon,
        forecast_horizon=forecast_horizon,
        normalize=True,
    )

    if train:
        forecaster = Forecaster(
            input_dim=history_horizon,
            output_dim=forecast_horizon,
            num_layers=num_layers,
            num_hidden=num_hidden,
        )

        trainer = IndependentTrainer(forecaster, training_dataset)

        trainer.train(
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            save_dir=save_dir,
        )
    else:
        forecaster = Forecaster(
            input_dim=history_horizon,
            output_dim=forecast_horizon,
            num_layers=num_layers,
            num_hidden=num_hidden,
            load_dir=load_dir,
        )

    # Testing model
    testing_dataset = ForecastDataset(
        data_dir=testing_dir,
        column=column,
        history_horizon=history_horizon,
        forecast_horizon=forecast_horizon,
        normalize=False,
    )

    # Compute metrics
    mse_mean, mse_std = compute_prediction_metric(
        forecaster, training_dataset, testing_dataset, squared_error
    )

    mae_mean, mae_std = compute_prediction_metric(
        forecaster, training_dataset, testing_dataset, absolute_error
    )

    capex_mean, capex_std = compute_capex_metric(
        forecaster, ed_model, training_dataset, testing_dataset, capex, lambdas
    )

    print(f"MSE Mean: {mse_mean.item()} - STD: {mse_std.item()}")
    print(f"MAE Mean: {mae_mean.item()} - STD: {mae_std.item()}")
    print(f"CAPEX Mean: {capex_mean.item()} - STD: {capex_std.item()}")

    # Visualize some predictions
    for i in np.random.randint(0, len(testing_dataset), 15):
        x, y = testing_dataset.__getitem__(i)

        x = (x - training_dataset.normalization_min) / (
            training_dataset.normalization_max - training_dataset.normalization_min
        )

        y_hat = forecaster(torch.from_numpy(x)).detach().numpy()

        y_hat = (
            y_hat
            * (training_dataset.normalization_max - training_dataset.normalization_min)
            + training_dataset.normalization_min
        )

        plt.plot(y_hat, label="Prediction")
        plt.plot(y, label="Ground Truth")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    independent_model()
