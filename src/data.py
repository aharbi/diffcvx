import numpy as np
import argparse
import pandas as pd
import gridstatus

from torch.utils.data import Dataset

from numpy.lib.stride_tricks import sliding_window_view


class ForecastDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        column: str,
        history_horizon: int,
        forecast_horizon: int,
    ):
        """Generates a PyTorch dataset for a forecasting problem.

        Args:
            data_dir (str): directory where time series dataframe is located.
            column (str): dataframe column for the time series.
            history_horizon (int): length of the prediction history window.
            forecast_horizon (int): length of the prediction future window.
        """
        self.data_dir = data_dir
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon

        self.data = pd.read_csv(data_dir)

        self.time_series = self.data[column].to_numpy()

        self.time_series_shifted = self.time_series[history_horizon:]

        self.time_series = self.time_series.astype(np.float32)
        self.time_series_shifted = self.time_series_shifted.astype(np.float32)

        # X, and Y have shape (number of examples, length of horizon window)
        self.X = sliding_window_view(self.time_series, history_horizon)
        self.Y = sliding_window_view(self.time_series_shifted, forecast_horizon)

        self.n = self.Y.shape[0]

        # Match number of examples in X and Y.
        self.X = self.X[: self.n, :]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx, :]

        return x, y


def get_caiso(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    save_dir: str,
    normalization: float,
):
    """Generates hourly electrical load data from the CAISO network.

    Args:
        start_date (pandas.Timestamp): start date of generated data.
        end_date (pandas.Timestamp): end date of generated data.
        save_dir (str): path to the directory where the generated csv will be saved.
        normalization (float): value to normalize the time series with.

    Returns:
        pandas.Dataframe: dataframe of hourly electrical load.
    """
    caiso = gridstatus.CAISO()

    data = caiso.get_load(start=start_date, end=end_date)

    load = data.set_index("Time").resample("h").mean()
    load.drop([load.columns[0], load.columns[1]], axis=1, inplace=True)
    load["Load"] = load["Load"] / normalization

    load.dropna(inplace=True)

    load.to_csv(save_dir)

    return load


def get_data(args):
    # Training data
    train_start_date = pd.Timestamp(args.train_start_date).normalize()
    train_end_date = pd.Timestamp(args.train_end_date).normalize()
    train_save_dir = args.save_dir + "_train.csv"
    normalization = args.normalization

    _ = get_caiso(train_start_date, train_end_date, train_save_dir, normalization)

    # Testing data
    test_start_date = pd.Timestamp(args.test_start_date).normalize()
    test_end_date = pd.Timestamp(args.test_end_date).normalize()
    test_save_dir = args.save_dir + "_test.csv"

    _ = get_caiso(test_start_date, test_end_date, test_save_dir, normalization)


def get_args():
    parser = argparse.ArgumentParser(description="download CAISO data")

    parser.add_argument(
        "--train_start_date",
        metavar="train_start_date",
        type=str,
        default="January 1, 2023",
        help="enter the start date for training data (e.g., January 1, 2023)",
    )
    parser.add_argument(
        "--train_end_date",
        metavar="train_end_date",
        type=str,
        default="February 28, 2023",
        help="enter the start date for training data (e.g., January 1, 2023)",
    )
    parser.add_argument(
        "--test_start_date",
        metavar="test_start_date",
        type=str,
        default="March 1, 2023",
        help="enter the start date for testing data (e.g., January 1, 2023)",
    )
    parser.add_argument(
        "--test_end_date",
        metavar="test_end_date",
        type=str,
        default="April 30, 2023",
        help="enter the start date for testing data (e.g., January 1, 2023)",
    )
    parser.add_argument(
        "--save_dir",
        metavar="save_dir",
        type=str,
        default="data/caiso.csv",
        help="enter location where to save the data",
    )
    parser.add_argument(
        "--normalization",
        metavar="normalization",
        type=float,
        default=50000,
        help="enter a value to normalize the data with",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    get_data(args)
