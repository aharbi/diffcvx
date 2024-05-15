import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        normalize: bool,
    ):
        """Generates a PyTorch dataset for a forecasting problem.

        Args:
            data_dir (str): directory where time series dataframe is located.
            column (str): dataframe column for the time series.
            history_horizon (int): length of the prediction history window.
            forecast_horizon (int): length of the prediction future window.
            normalize (bool): whether to normalize the data to (0, 1) or not.
        """
        self.data_dir = data_dir
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon

        self.data = pd.read_csv(data_dir)

        self.time_series = self.data[column].to_numpy()

        self.normalization_min = self.time_series.min()
        self.normalization_max = self.time_series.max()

        if normalize:
            self.time_series = (self.time_series - self.time_series.min()) / (
                self.time_series.max() - self.time_series.min()
            )

        self.time_series_shifted = self.time_series[history_horizon:]

        self.time_series = self.time_series.astype(np.float32)
        self.time_series_shifted = self.time_series_shifted.astype(np.float32)

        # X, and Y have shape (number of examples, length of horizon window)
        self.X = sliding_window_view(self.time_series, history_horizon)
        self.Y = sliding_window_view(self.time_series_shifted, forecast_horizon)

        self.n = self.Y.shape[0]

        # Match number of examples in X and Y.
        self.X = self.X[: self.n, :]

    def get_normalization_constants(self):
        return (self.normalization_min, self.normalization_max)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx, :]

        return x, y


def get_caiso(start_date, end_date, save_dir):
    """Generates hourly electrical load data from the CAISO network.

    Args:
        start_date (pandas.Timestamp): start date of generated data.
        end_date (pandas.Timestamp): end date of generated data.
        save_dir (str): path to the directory where the generated csv will be saved.

    Returns:
        pandas.Dataframe: dataframe of hourly electrical load.
    """
    caiso = gridstatus.CAISO()

    data = caiso.get_load(start=start_date, end=end_date)

    load = data.set_index("Time").resample("h").mean()
    load.drop([load.columns[0], load.columns[1]], axis=1, inplace=True)
    load.dropna(inplace=True)

    load.to_csv(save_dir)

    return load
