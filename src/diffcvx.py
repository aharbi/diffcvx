import argparse
import json

from forecast import *
from model import *
from train import *
from metrics import *


def run_model(args, baseline=False):
    # Initialize ED model
    power_system_specs = json.load(open(args.system_dir))
    generators = [Generator(specification) for specification in power_system_specs]

    ed_model = EconomicDispatchModel(
        generators=generators,
        horizon=args.forecast_horizon,
        lambda_plus=args.lambda_plus,
        lambda_minus=args.lambda_minus,
    )

    # Load datasets
    column = "Load"

    training_dataset = ForecastDataset(
        data_dir=args.training_dir,
        column=column,
        history_horizon=args.history_horizon,
        forecast_horizon=args.forecast_horizon,
    )

    testing_dataset = ForecastDataset(
        data_dir=args.testing_dir,
        column=column,
        history_horizon=args.history_horizon,
        forecast_horizon=args.forecast_horizon,
    )

    # Train model
    if args.train:
        forecaster = Forecaster(
            input_dim=args.history_horizon,
            output_dim=args.forecast_horizon,
            num_layers=args.num_layers,
            num_hidden=args.num_hidden,
            load_dir=args.load_dir,
        )

        if baseline == False:
            trainer = EndToEndTrainer(forecaster, ed_model, training_dataset)

            trainer.train(
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                save_dir=args.save_dir,
                name=args.name,
            )
        else:
            trainer = IndependentTrainer(forecaster, training_dataset)

            trainer.train(
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                save_dir=args.save_dir,
                name=args.name,
            )
    else:
        forecaster = Forecaster(
            input_dim=args.history_horizon,
            output_dim=args.forecast_horizon,
            num_layers=args.num_layers,
            num_hidden=args.num_hidden,
            load_dir=args.load_dir,
        )

    # Test model
    mse_mean, mse_std = compute_prediction_metric(
        forecaster, testing_dataset, squared_error
    )

    mae_mean, mae_std = compute_prediction_metric(
        forecaster, testing_dataset, absolute_error
    )

    capex_mean, capex_std = compute_capex(
        forecaster, ed_model, testing_dataset, args.lambda_plus, args.lambda_minus
    )

    ramping_mean, ramping_std = compute_ramping_reserve(
        forecaster, ed_model, testing_dataset
    )

    print(f"MSE Mean: {mse_mean.item()} - STD: {mse_std.item()}")
    print(f"MAE Mean: {mae_mean.item()} - STD: {mae_std.item()}")
    print(f"CAPEX Mean: {capex_mean.item()} - STD: {capex_std.item()}")
    print(f"Ramping Mean: {ramping_mean.item()} - STD: {ramping_std.item()}")


def get_args():
    parser = argparse.ArgumentParser(description="download CAISO data")

    parser.add_argument(
        "--history_horizon",
        metavar="history_horizon",
        type=int,
        default=24,
        help="",
    )

    parser.add_argument(
        "--forecast_horizon",
        metavar="forecast_horizon",
        type=int,
        default=24,
        help="",
    )

    parser.add_argument(
        "--loss",
        metavar="loss",
        type=str,
        default="capex",
        help="",
    )

    parser.add_argument(
        "--num_layers",
        metavar="num_layers",
        type=int,
        default=1,
        help="",
    )

    parser.add_argument(
        "--train",
        metavar="train",
        type=bool,
        default=True,
        help="",
    )

    parser.add_argument(
        "--num_hidden",
        metavar="num_hidden",
        type=int,
        default=32,
        help="",
    )

    parser.add_argument(
        "--num_epochs",
        metavar="num_epochs",
        type=int,
        default=15,
        help="",
    )

    parser.add_argument(
        "--batch_size",
        metavar="batch_size",
        type=int,
        default=8,
        help="",
    )

    parser.add_argument(
        "--lr",
        metavar="lr",
        type=float,
        default=1e-2,
        help="",
    )

    parser.add_argument(
        "--device",
        metavar="device",
        type=str,
        default="cpu",
        help="",
    )

    parser.add_argument(
        "--lambda_plus",
        metavar="lambda_plus",
        type=float,
        default=1000,
        help="",
    )

    parser.add_argument(
        "--lambda_minus",
        metavar="lambda_minus",
        type=float,
        default=5000,
        help="",
    )

    parser.add_argument(
        "--training_dir",
        metavar="training_dir",
        type=str,
        default="data/caiso_train.csv",
        help="",
    )

    parser.add_argument(
        "--testing_dir",
        metavar="testing_dir",
        type=str,
        default="data/caiso_test.csv",
        help="",
    )

    parser.add_argument(
        "--system_dir",
        metavar="system_dir",
        type=str,
        default="data/system.json",
        help="",
    )

    parser.add_argument(
        "--system_dir",
        metavar="system_dir",
        type=str,
        default="data/system.json",
        help="",
    )

    parser.add_argument(
        "--save_dir",
        metavar="save_dir",
        type=str,
        default="models/",
        help="",
    )

    parser.add_argument(
        "--name",
        metavar="name",
        type=str,
        default="model",
        help="",
    )

    parser.add_argument(
        "--load_dir",
        metavar="load_dir",
        type=str,
        default="models/mlp_baseline_14.pth",
        help="",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    run_model(args, baseline=False)
