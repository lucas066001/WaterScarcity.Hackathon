import os
import re
import random
from math import sqrt
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .helpers import save_or_create
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from quantile_forest import RandomForestQuantileRegressor


def standardize_values(
    y: np.ndarray, stations: np.ndarray, station_stats: pd.DataFrame
) -> np.ndarray:
    """
    Standardize values based on station-level statistics.

    Parameters:
        y (np.ndarray): The values to standardize.
        stations (np.ndarray): The station codes.
        station_stats (pd.DataFrame): The station-level statistics.

    Returns:
        np.ndarray: The standardized values.
    """
    out = np.empty_like(y, dtype=float)
    for s in np.unique(stations):
        idx = stations == s
        min = station_stats.loc[s, "min"]
        max = station_stats.loc[s, "max"]
        out[idx] = (y[idx] - min) * 100.0 / (max - min)
    return out


def split_dataset(
    ds: pd.DataFrame,
    p: float = 0.75,
    time: str = None,
    seed: int = 42,
    test_stations: np.ndarray = None,
):
    """
    Splits the dataset into training and testing sets
    based on the specified method.

    Parameters:
        ds (pd.DataFrame): The dataset containing data.
        p (float): The proportion of stations
        for the training set.
        time (str, optional): The timestamp for temporal split.
        seed (int): Random seed for reproducibility (default is 42).
        test_stations (np.ndarray): The list of test stations
        (for the Spatio-Temporal split).

    Returns:
        tuple: X_train, y_train, X_test, y_test, train_stations, test_stations
    """
    station_code = ds["station_code"].unique()
    random.seed(seed)
    random.shuffle(station_code)

    if test_stations is not None:
        train_stations = [s for s in station_code if s not in test_stations]
    else:
        test_stations = station_code[int(len(station_code) * p) :]
        train_stations = station_code[: int(len(station_code) * p)]

    test_temporal = ds[pd.to_datetime(ds.index) >= pd.to_datetime(time)]
    test_temporal = test_temporal[test_temporal["station_code"].isin(train_stations)]

    train = ds[ds["station_code"].isin(train_stations)]
    test_spatio_temporal = ds[ds["station_code"].isin(test_stations)]
    train = train[pd.to_datetime(train.index) < pd.to_datetime(time)]
    test_spatio_temporal = test_spatio_temporal[
        pd.to_datetime(test_spatio_temporal.index) >= pd.to_datetime(time)
    ]
    return train, test_spatio_temporal, test_temporal


def get_station_stats(y: np.ndarray, station_code: np.ndarray) -> pd.DataFrame:
    """
    Compute station-level statistics for the given data.

    Args:
        y (np.ndarray): A NumPy array of numeric measurements or target values.
        station_code (np.ndarray): A NumPy array of station codes.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds
            to a unique station code and includes statistics.
    """
    df = pd.DataFrame({"y": y, "station_code": station_code})
    station_stats = df.groupby("station_code")["y"].agg(["mean", "std", "min", "max"])
    return station_stats


def standardize_prediction_intervals(
    y_pred_intervals: np.ndarray, stations: np.ndarray, station_stats: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardizes the prediction interval values for a set of
    stations using the provided station statistics.

    Args:
        y_pred_intervals (np.ndarray): the predicted interval values.
        stations (np.ndarray): station codes.
        station_stats (pd.DataFrame): statistics for each station.

    Returns:
        Tuple[np.ndarray, np.ndarray]: standardized lower and
            upper prediction interval values.
    """
    if y_pred_intervals is None:
        return None, None

    if len(y_pred_intervals.shape) == 3:
        y_pred_lower_std = standardize_values(
            y_pred_intervals[:, 0, 0], stations, station_stats
        )
        y_pred_upper_std = standardize_values(
            y_pred_intervals[:, 1, 0], stations, station_stats
        )
    else:
        y_pred_lower_std = standardize_values(
            y_pred_intervals[:, 0], stations, station_stats
        )
        y_pred_upper_std = standardize_values(
            y_pred_intervals[:, 1], stations, station_stats
        )

    return y_pred_lower_std, y_pred_upper_std


def compute_per_station_metrics(
    y_true_std: np.ndarray,
    y_pred_std: np.ndarray,
    stations: np.ndarray,
    y_pred_lower_std: np.ndarray = None,
    y_pred_upper_std: np.ndarray = None,
) -> pd.DataFrame:
    """
    Compute station-level performance metrics including scaled RMSE,
    scaled MAE, coverage, scaled prediction interval size,
    and Gaussian negative log-likelihood.

    Parameters:
        y_true_std (np.ndarray): Standardized ground truth.
        y_pred_std (np.ndarray): Array of predicted standardized predictions.
        stations (np.ndarray): Station codes.
        y_pred_lower_std (np.ndarray): lower prediction interval values.
        y_pred_upper_std (np.ndarray): upper prediction interval values.

    Returns:
    pd.DataFrame
        Dataframe with the following metrics:
            - station_code: Identifier for the station.
            - scaled_rmse: Scaled Root Mean Squared Error for the station.
            - scaled_mae: Scaled Mean Absolute Error for the station.
            - coverage
            - scaled_interval_size: Average size of the prediction interval
            - log_likelihood: Gaussian negative log-likelihood.
    """
    station_list = np.unique(stations)

    records = []

    has_intervals = (y_pred_lower_std is not None) and (y_pred_upper_std is not None)

    for s in station_list:
        idx = stations == s
        y_true_s = y_true_std[idx]
        y_pred_s = y_pred_std[idx]

        rmse_s = sqrt(mean_squared_error(y_true_s, y_pred_s))
        mae_s = mean_absolute_error(y_true_s, y_pred_s)

        if has_intervals:
            y_lower_s = y_pred_lower_std[idx]
            y_upper_s = y_pred_upper_std[idx]

            # Estimate sigma using the 95% confidence interval approximation
            sigma_s = (y_upper_s - y_lower_s) / 3.29

            # Compute Gaussian negative log-likelihood
            nll_s = (1 / len(y_true_s)) * np.sum(
                np.log(sigma_s) + abs((y_true_s - y_pred_s)) / abs(2 * sigma_s)
            )

            coverage_s = np.mean((y_true_s >= y_lower_s) & (y_true_s <= y_upper_s))
            interval_size_s = np.mean(y_upper_s - y_lower_s)
        else:
            sigma_s = np.std(y_true_s - y_pred_s)  # Fallback estimation
            sigma_s = max(sigma_s, 1e-6)  # Ensure non-zero, positive sigma

            nll_s = (1 / len(y_true_s)) * np.sum(
                np.log(sigma_s) + abs((y_true_s - y_pred_s)) / abs(2 * sigma_s)
            )

            coverage_s = np.nan
            interval_size_s = np.nan

        # Collect station-level metrics
        records.append(
            {
                "station_code": s,
                "scaled_rmse": rmse_s,
                "scaled_mae": mae_s,
                "coverage": coverage_s,
                "scaled_interval_size": interval_size_s,
                "log_likelihood": nll_s,
            }
        )

    return pd.DataFrame(records)


def summarize_metrics(
    metrics: pd.DataFrame, model_name: str, dataset_type: str
) -> pd.DataFrame:
    """
    Given a station-level metrics DataFrame, compute average (per station)
    values and a final score.

    Parameters:
        metrics (pd.DataFrame): station-level metrics.
        model_name (str): The name of the model.
        dataset_type (str): The type of dataset used (e.g., "test").

    Returns:
    pd.DataFrame
        A DataFrame containing the final model-level metrics
        (scaled RMSE, log-likelihood, scaled MAE, coverage,
        scaled interval size).
    """
    rmse_final = np.nanmean(metrics["scaled_rmse"])
    mae_final = np.nanmean(metrics["scaled_mae"])
    log_likelihood = np.nanmean(metrics["log_likelihood"])
    if metrics["coverage"].count() == 0:
        coverage_final = np.nan
        interval_size_final = np.nan
    else:
        coverage_final = np.nanmean(metrics["coverage"])
        interval_size_final = np.nanmean(metrics["scaled_interval_size"])

    data = {
        "model": [model_name],
        "dataset": [dataset_type],
        "scaled_rmse": [rmse_final],
        "log_likelihood": [log_likelihood],
        "scaled_mae": [mae_final],
        "coverage": [coverage_final],
        "scaled_interval_size": [interval_size_final],
    }
    return pd.DataFrame(data)


def print_summary_table(summary_df: pd.DataFrame):
    """
    Print a summary of the model-level metrics using tabulate.
    """
    row = summary_df.iloc[0]
    table_data = [
        ["Model", row["model"]],
        ["Dataset Type", row["dataset"]],
        ["Average Scaled RMSE", row["scaled_rmse"]],
        ["Average Scaled MAE", row["scaled_mae"]],
        ["Average Coverage", row["coverage"]],
        ["Average Scaled Interval Size", row["scaled_interval_size"]],
        ["Final Score", row["final_score"]],
    ]
    print(
        "\n"
        + tabulate(table_data, headers=["Metric", "Value"], tablefmt="pretty")
        + "\n"
    )


def generate_boxplots(
    station_metrics_df: pd.DataFrame,
    column_to_display: str,
    prefix: str,
    title: str,
    save: bool = False,
    display: bool = True,
):
    """
    Generate and save boxplots based on the station-level metrics.
    If RMSE_mode is True, only shows a scaled RMSE boxplot.
    Otherwise, shows coverage & interval size boxplots.
    """
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    station_metrics_df.boxplot(column=column_to_display, by="model", ax=ax1)
    ax1.set_title("Per-Station Scaled Gaussian Log-Likelihood")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Scaled GLL")
    fig1.suptitle(title)
    plt.tight_layout()
    if display:
        plt.show()

    if save:
        current_date = pd.Timestamp.now().strftime("%d-%m-%Y_%H-%M")
        path = f"../../figures/models/{prefix}_{current_date}_boxplots.png"
        save_or_create(plt, path)


def compare_models_per_station(
    y: np.ndarray,
    predictions: List[dict],
    station_code: np.ndarray,
    prefix: str = "",
    column_to_display: str = True,
    title: str = "Model Evaluation",
    save: bool = False,
    display: bool = True,
    return_df: bool = False,
):
    """
    Evaluate the performance of one or multiple models at the station level.
    Scaled interval size for each station.

    Parameters:
        y : np.ndarray
            Ground truth values for the test set.
        predictions : List[dict]
            A list of prediction dictionaries. Each dictionary must include:
            - "model": A string with the model name.
            - "dataset": Either "train" or "test".
            - "prediction": A 1D array of predicted values.
            - "prediction_interval" (optional): interval bounds.
        station_code : np.ndarray
            An array of station identifiers corresponding to each entry in y.
        prefix : str, optional
            A string prefix for saving figures.
        column_to_display : str, optional
            The column to display in the boxplots.
        title : str, optional
            The title of the boxplots.
        save : bool, optional
            If True, the boxplots are saved.
        display : bool, optional
            If True, the boxplots are displayed.
    """
    all_station_metrics = []

    for pred in predictions:
        model_name = pred["model"]

        station_stats = get_station_stats(y, station_code)

        y_pred = pred["prediction"]
        y_pred_intervals = pred.get("prediction_interval", None)

        y_true_std = standardize_values(y, station_code, station_stats)
        y_pred_std = standardize_values(y_pred, station_code, station_stats)

        y_pred_lower_std, y_pred_upper_std = standardize_prediction_intervals(
            y_pred_intervals, station_code, station_stats
        )

        station_metrics_df = compute_per_station_metrics(
            y_true_std=y_true_std,
            y_pred_std=y_pred_std,
            stations=station_code,
            y_pred_lower_std=y_pred_lower_std,
            y_pred_upper_std=y_pred_upper_std,
        )

        station_metrics_df["model"] = model_name
        all_station_metrics.append(station_metrics_df)

    station_metrics_df = pd.concat(all_station_metrics, ignore_index=True)

    if display:
        generate_boxplots(
            station_metrics_df, column_to_display, prefix, title, save, display
        )

    if return_df:
        return station_metrics_df


def load_models_auto(mn: str, dir: str = "../../models/") -> List[any]:
    """
    Auto-load the latest models
    for week0, week1, and week2 from the specified directory.

    Parameters:
        mn (str): The base model name to search for
        (e.g., "mapie_quantile").
        dir (str): Directory where models are stored.

    Returns:
    - List of loaded models in the order [week0, week1, week2].
    """

    p = rf"^{mn}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})_week_([0-9]).pkl$"
    pattern = re.compile(p)
    latest_paths = {}

    for fname in os.listdir(dir):
        match = pattern.match(fname)
        if match:
            date_str, week_str = match.groups()
            week_num = int(week_str)
            date_obj = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
            if week_num not in latest_paths:
                latest_paths[week_num] = (date_obj, fname)
            else:
                current_latest_date, _ = latest_paths[week_num]
                if date_obj > current_latest_date:
                    latest_paths[week_num] = (date_obj, fname)

    loaded_mapie = []
    for i in [0, 1, 2, 3]:
        if i not in latest_paths:
            raise ValueError(f"No mapie_quantile model found for week{i} in {dir}.")
        model_file = latest_paths[i][1]
        full_path = os.path.join(dir, model_file)
        print(f"Loading model for week {i}: {model_file}")
        loaded_mapie.append(joblib.load(full_path))

    return loaded_mapie


def custom_log_likelihood(estimator, X, y_true, cv_data, station_stats, alpha=0.1):
    """
    Custom log-likelihood scoring function.

    Parameters:
        estimator : The fitted estimator with a .predict method.
        X : DataFrame of predictor variables.
        y_true : True target values.
        cv_data : Full DataFrame that includes extra columns
        (e.g., "station_code").
        station_stats : Station-level statistics needed for standardization.
        alpha : Significance level (default from ALPHA).

    Returns:
        nll_s : Computed log-likelihood score.
    """
    # Align y_true with X.
    y_true = pd.Series(y_true.values, index=X.index)

    # Get predictions.
    y_pred = estimator.predict(X)

    # Get quantile predictions.
    y_quantiles = estimator.predict(X, quantiles=[alpha / 2, 1 - alpha / 2])

    # Retrieve station codes from cv_data using X's indices.
    current_stations = cv_data.loc[X.index, "station_code"].to_numpy()

    # Standardize the values.
    y_true_std = standardize_values(y_true.to_numpy(), current_stations, station_stats)
    y_pred_std = standardize_values(y_pred, current_stations, station_stats)
    y_lower_std, y_upper_std = standardize_prediction_intervals(
        y_quantiles, current_stations, station_stats
    )

    # Compute sigma from the prediction interval.
    sigma_std = (y_upper_std - y_lower_std) / 3.29
    sigma_std = np.maximum(sigma_std, 1e-6)

    # Compute the negative log-likelihood.
    nll_s = (1 / len(y_true_std)) * np.sum(
        np.log(sigma_std) + np.abs(y_true_std - y_pred_std) / (2 * sigma_std)
    )

    # Optionally, print some diagnostics.
    cov = np.mean((y_true_std >= y_lower_std) & (y_true_std <= y_upper_std))
    i_size = np.mean(y_upper_std - y_lower_std)
    print(f"Fold: coverage = {cov:.3f}, interval size = {i_size:.3f}")

    return nll_s


def create_deep_model(input_shape: Tuple[int]):
    """Define a simple deep learning model for regression."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(1, activation="linear"),  # regression output
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


import numpy as np
from xgboost import DMatrix, XGBRegressor, train
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class XGBQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Custom XGBoost model trained on log(y) to ensure positive predictions.
    """

    def __init__(
        self,
        quantile=0.5,
        n_estimators=500,
        learning_rate=0.1,
        max_depth=3,
        objective="reg:squaredlogerror",
        gamma=0,
        min_child_weight=1,
        subsample=1,
        colsample_bytree=1,
    ):
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.objective = objective
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.model_ = None

    def _quantile_gradient(self, y_true, y_pred):
        """Gradient of quantile loss."""
        residual = y_true - y_pred
        return np.where(residual > 0, -self.quantile, -(self.quantile - 1))

    def _quantile_hessian(self, y_true):
        """Hessian (second derivative) is constant for quantile loss."""
        return np.ones_like(y_true)

    def fit(self, X, y, eval_set=None):
        """Train model on log-transformed target."""
        feature_names = X.columns.tolist()

        X, y = check_X_y(X, y)

        if np.any(y <= 0):
            raise ValueError(
                "All target values must be positive for log transformation."
            )

        y_log = np.log(y)  # Apply log transformation

        dtrain = DMatrix(X, label=y_log, feature_names=feature_names)

        params = {
            "objective": self.objective,
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }

        if eval_set is not None:
            eval_set = [
                (DMatrix(X_eval, label=np.log(y_eval)), "validation")
                for X_eval, y_eval in eval_set
            ]

        self.model_ = train(
            params,
            dtrain,
            evals=eval_set,
            num_boost_round=self.n_estimators,
            obj=self._custom_loss,
            verbose_eval=False,
        )
        return self

    def _custom_loss(self, y_pred, dtrain):
        """Custom loss function."""
        y_true = dtrain.get_label()
        grad = self._quantile_gradient(y_true, y_pred)
        hess = self._quantile_hessian(y_true)
        return grad, hess

    def predict(self, X):
        """Predict and exponentiate to ensure positive values."""
        check_is_fitted(self, "model_")
        X = check_array(X)

        dtest = DMatrix(X)
        y_pred_log = self.model_.predict(dtest, validate_features=False)
        return np.exp(y_pred_log)  # Convert back to original scale


class XGBQRFModel:
    def __init__(
        self, xgb_params: dict, qrf_params: dict, quantiles: list = [], random_state=42
    ):
        self.xgb_params = xgb_params
        self.qrf_params = qrf_params
        self.quantiles = quantiles
        self.models = {
            "XGB": {
                q: XGBQuantileRegressor(quantile=q, **self.xgb_params)
                for q in self.quantiles
            },
            "QRF": RandomForestQuantileRegressor(**self.qrf_params),
        }

    def fit(self, X, y, eval_set: list = []):
        print("Fitting XGB models")
        for q in self.quantiles:
            self.models["XGB"][q].fit(X, y, eval_set=eval_set)
        print("Fitting QRF model")
        self.models["QRF"].fit(X, y)

    def predict(self, X):
        xgb_predictions = np.stack(
            [model.predict(X) for model in self.models["XGB"].values()], axis=1
        )

        qrf_predictions = self.models["QRF"].predict(
            X, quantiles=self.quantiles, aggregate_leaves_first=False
        )

        lower_gap = qrf_predictions[:, 1] - qrf_predictions[:, 0]
        lower = xgb_predictions[:, 1] - lower_gap
        lower = lower.clip(min=0)

        upper_gap_qrf = qrf_predictions[:, 2] - qrf_predictions[:, 1]
        upper = xgb_predictions[:, 1] + upper_gap_qrf
        return np.stack(
            [lower, xgb_predictions[:, 1], upper],
            axis=1,
        )


from xgboost import XGBRegressor


class XGBQRF_SimpleModel:
    def __init__(
        self, xgb_params: dict, qrf_params: dict, quantiles: list = [], SEED: int = 42
    ):
        self.xgb_params = xgb_params
        self.qrf_params = qrf_params
        self.quantiles = quantiles
        self.models = {
            "XGB": XGBRegressor(**self.xgb_params, random_state=SEED),
            "QRF": RandomForestQuantileRegressor(**self.qrf_params, random_state=SEED),
        }
        self.predictions = None

    def fit(self, X, y, eval_set: list = [], prev_week_models=[]):
        week = len(prev_week_models)
        if len(prev_week_models) > 0:
            for model in prev_week_models:
                X = pd.concat(
                    [
                        X.reset_index(drop=True),
                        model.predictions.reset_index(drop=True),
                    ],
                    axis=1,
                )
        self.models["XGB"].fit(X, y, eval_set=eval_set, verbose=False)
        print("Fitting QRF model")
        self.models["QRF"].fit(X, y)

        self.save_predictions(X, week)

    def save_predictions(self, X, week, predictions=None):
        if predictions is None:
            predictions = self.predict(X)

        predictions = pd.DataFrame(
            {
                f"predicted_water_flow_week{week}_lower": predictions[:, 0],
                f"predicted_water_flow_week{week}_median": predictions[:, 1],
                f"predicted_water_flow_week{week}_upper": predictions[:, 2],
            }
        )

        predictions.set_index(X.index)
        self.predictions = predictions

    def predict(self, X, prev_week_models=[]):
        week = len(prev_week_models)
        if len(prev_week_models) > 0:
            for model in prev_week_models:
                X = pd.concat(
                    [
                        X.reset_index(drop=True),
                        model.predictions.reset_index(drop=True),
                    ],
                    axis=1,
                )
        xgb_predictions = self.models["XGB"].predict(X)

        qrf_predictions = self.models["QRF"].predict(X, quantiles=self.quantiles)

        qrf_median = qrf_predictions[:, 1]

        # Threshold on xgb_predictions
        xgb_predictions[xgb_predictions < 100] = qrf_median[xgb_predictions < 100]

        qrf_lower_gap = (
            qrf_median - qrf_predictions[:, 0]
        ) / qrf_median  # à soustraire
        lower_bound = xgb_predictions * (1 - qrf_lower_gap)

        qrf_upper_gap = (qrf_predictions[:, 2] - qrf_median) / qrf_median  # à ajouter
        upper_bound = xgb_predictions * (1 + qrf_upper_gap)

        predictions = np.stack(
            [lower_bound, xgb_predictions, upper_bound],
            axis=1,
        )

        self.save_predictions(X, week, predictions)

        return predictions

    def predict_separate(self, X):
        predictions = dict()
        predictions["XGB"] = self.models["XGB"].predict(X)
        predictions["QRF"] = self.models["QRF"].predict(X, quantiles=self.quantiles)
        return predictions


class Ensemble:
    def __init__(
        self,
        params: dict = {},
        quantiles: list[float] = [0.05, 0.5, 0.95],
        seed: int = 42,
    ):
        self.params = params

        if len(self.params.keys()) == 0:
            print("No parameters provided for the models using default ones.")
            self.xgb_params = {
                i: {
                    "n_estimators": 300,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                }
                for i in range(3)
            }

            self.qrf_params = {
                i: {
                    "n_estimators": 30,
                    "max_depth": 10,
                }
                for i in range(3)
            }
        if len(self.params.keys()) == 3:
            self.xgb_params = {
                k: {
                    "n_estimators": v["n_estimators"],
                    "max_depth": v["max_depth"],
                    "learning_rate": v["learning_rate"],
                    "subsample": v["subsample"],
                }
                for k, v in self.params.items()
            }

            self.qrf_params = {
                k: {
                    "n_estimators": v["n_estimators_qrf"],
                    "max_depth": v["max_depth_qrf"],
                    "min_samples_split": v["min_samples_split"],
                    "min_samples_leaf": v["min_samples_leaf"],
                    "criterion": v["criterion"],
                }
                for k, v in self.params.items()
            }

        self.models = {
            k: XGBQRF_SimpleModel(
                xgb_params=self.xgb_params[k],
                qrf_params=self.qrf_params[k],
                quantiles=quantiles,
                SEED=seed,
            )
            for k in range(3)
        }

    def fit(self, X, y, eval_set: list = [], X_qrf=None, y_qrf=None):
        for i in self.models.keys():
            print(f"Training model {i}")
            if i == 0:
                # Brazil dedicated
                x_brazil = X[X["north_hemisphere"] == 0]
                y_brazil = y[X["north_hemisphere"] == 0]
                x_brazil_qrf = None
                y_brazil_qrf = None
                if not X_qrf is None and not y_qrf is None:
                    x_brazil_qrf = X_qrf[X_qrf["north_hemisphere"] == 0]
                    y_brazil_qrf = y_qrf[X_qrf["north_hemisphere"] == 0]
                # brazil_eval =
                self.models[i].fit(
                    x_brazil,
                    y_brazil,
                    eval_set=eval_set,
                    # X_qrf=x_brazil_qrf,
                    # y_qrf=y_brazil_qrf,
                )
            elif i == 1:
                x_north = X[X["north_hemisphere"] == 1]
                y_north = y[X["north_hemisphere"] == 1]
                x_north_qrf = None
                y_north_qrf = None
                if not X_qrf is None and not y_qrf is None:
                    x_north_qrf = X_qrf[X_qrf["north_hemisphere"] == 1]
                    y_north_qrf = y_qrf[X_qrf["north_hemisphere"] == 1]
                self.models[i].fit(
                    x_north,
                    y_north,
                    eval_set=eval_set,
                    # X_qrf=x_north_qrf,
                    # y_qrf=y_north_qrf,
                )
            else:
                self.models[i].fit(X, y, eval_set=eval_set)

    def predict(self, X):
        predictions = []
        for i in self.models.keys():
            pred = self.models[i].predict(X)
            predictions.append(pred)

        predictions = np.stack(predictions, axis=1)
        weights = np.ones((X.shape[0], len(self.models)))

        north = X["north_hemisphere"].values  # shape (n_samples,)

        # Dynamic weights based on 'north_hemisphere'
        weights[:, 0] = np.where(north == 0, 0.6, 0.0)
        weights[:, 1] = np.where(north == 1, 0.6, 0.0)
        weights[:, 2] = 0.4  # General model

        weights /= weights.sum(axis=1, keepdims=True)  # (N, 3)

        # Multiply weights with predictions
        # predictions: (N, 3, Q) * weights[:, :, None] → (N, 3, Q)
        weighted_predictions = predictions * weights[:, :, None]

        # Sum over models: (N, Q)
        final_prediction = np.sum(weighted_predictions, axis=1)

        return final_prediction


class ChainedQrfModel:
    def __init__(self, qrf_params: dict, qrf_features: dict, number_of_weeks: int = 4):
        self.qrf_params = qrf_params
        self.qrf_features = qrf_features
        self.number_of_weeks = number_of_weeks
        self.models = {}
        for i in range(self.number_of_weeks):
            self.models[i] = RandomForestQuantileRegressor(**self.qrf_params[i])

    def fit(self, X, y):
        print("Fitting QRF models")
        X_incremental = {}
        for i in range(self.number_of_weeks):
            print(f"Fitting week {i}")
            if i == 0:
                X_incremental[i] = X.copy(deep=True)
                self.models[i].fit(X_incremental[i][self.qrf_features[i]], y[i])
            else:
                # Use the previous week's predictions as features
                y_pred = self.models[i - 1].predict(
                    X_incremental[i - 1][self.qrf_features[i - 1]], quantiles=[0.5]
                )
                y_pred = np.reshape(y_pred, (-1, 1))

                X_incremental[i] = X_incremental[i - 1].copy(deep=True)
                print(f"week_{i}_pred")
                X_incremental[i][f"week_{i}_pred"] = y_pred

                if i == 1:
                    X_incremental[i][f"week_{i-1}_{i}_slope"] = (
                        X_incremental[i][f"week_{i}_pred"]
                        - X_incremental[i]["water_flow_lag_1w"]
                    ) / X_incremental[i]["water_flow_lag_1w"].replace(0, np.nan)
                else:
                    X_incremental[i][f"week_{i-1}_{i}_slope"] = (
                        X_incremental[i][f"week_{i}_pred"]
                        - X_incremental[i][f"week_{i-1}_pred"]
                    ) / X_incremental[i][f"week_{i-1}_pred"].replace(0, np.nan)

                self.models[i].fit(X_incremental[i][self.qrf_features[i]], y[i])

    def predict(self, X, quantiles=[0.05, 0.5, 0.95]):
        print("Predicting QRF models")
        predictions = {}
        X_incremental = {}
        for i in range(self.number_of_weeks):
            if i == 0:
                X_incremental[i] = X.copy(deep=True)
                predictions[i] = self.models[i].predict(
                    X_incremental[i][self.qrf_features[i]], quantiles=quantiles
                )
            else:
                # Use the previous week's predictions as features
                y_pred = predictions[i - 1][:, 1]
                y_pred = np.reshape(y_pred, (-1, 1))
                X_incremental[i] = X_incremental[i - 1].copy(deep=True)
                X_incremental[i][f"week_{i}_pred"] = y_pred

                if i == 1:
                    X_incremental[i][f"week_{i-1}_{i}_slope"] = (
                        X_incremental[i][f"week_{i}_pred"]
                        - X_incremental[i]["water_flow_lag_1w"]
                    ) / X_incremental[i]["water_flow_lag_1w"].replace(0, np.nan)
                else:
                    X_incremental[i][f"week_{i-1}_{i}_slope"] = (
                        X_incremental[i][f"week_{i}_pred"]
                        - X_incremental[i][f"week_{i-1}_pred"]
                    ) / X_incremental[i][f"week_{i-1}_pred"].replace(0, np.nan)

                predictions[i] = self.models[i].predict(
                    X_incremental[i][self.qrf_features[i]], quantiles=quantiles
                )

        return predictions


class SpecialistQrfModel:
    def __init__(
        self,
        qrf_params: dict,
        qrf_features: dict,
        specialized_col="region_cluster",
        number_of_weeks: int = 4,
        number_of_clusters: int = 3,
    ):
        self.qrf_params = qrf_params
        self.qrf_features = qrf_features
        self.number_of_weeks = number_of_weeks
        self.specialized_col = specialized_col
        self.number_of_clusters = number_of_clusters
        self.models = {}

        print("Init QRF models")
        for i in range(self.number_of_weeks):
            self.models[i] = {}
            for clust_index in range(self.number_of_clusters):
                self.models[i][clust_index] = RandomForestQuantileRegressor(
                    **self.qrf_params[clust_index]
                )

    def fit(self, X, y):
        print("Fitting QRF models")
        for i in range(self.number_of_weeks):
            print(f"Fitting week {i}")
            for clust_index in range(self.number_of_clusters):
                print(f"Fitting cluster {clust_index}")
                X["y_true"] = y[i]
                X_train = X[X[self.specialized_col] == clust_index].copy(deep=True)
                y_train = X_train["y_true"]
                X.drop(columns=["y_true"])
                X_train.drop(columns=["y_true"])
                self.models[i][clust_index].fit(
                    X_train[self.qrf_features[clust_index]], y_train
                )

    def predict(self, X, quantiles=[0.05, 0.5, 0.95]):
        print("Predicting QRF models")
        predictions = {}

        for i in range(self.number_of_weeks):
            print(f"Predicting week {i}")
            # Crée un DataFrame vide avec les bons index et colonnes (quantiles)
            preds = pd.DataFrame(index=X.index, columns=quantiles, dtype=float)
            print(preds.head())
            for clust_index in range(self.number_of_clusters):
                print(f"Predicting cluster {clust_index}")
                cluster_mask = X[self.specialized_col] == clust_index
                X_pred = X[cluster_mask].copy(deep=True)

                # Prédictions QRF : shape (n_samples, len(quantiles))
                cluster_preds = self.models[i][clust_index].predict(
                    X_pred[self.qrf_features[clust_index]], quantiles=quantiles
                )

                # Place les prédictions aux bons indices
                print(X_pred.index.shape)
                print(cluster_preds.shape)
                print(preds.shape)
                preds.loc[cluster_mask, :] = cluster_preds

            predictions[i] = preds

        return predictions


from sklearn.base import BaseEstimator


class XGBQRFWrapper(BaseEstimator):
    def __init__(
        self,
        quantiles=[0.05, 0.5, 0.95],
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        n_estimators_qrf=100,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="squared_error",
        max_depth_qrf=5,
        random_state=42,
    ):
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.n_estimators_qrf = n_estimators_qrf
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_depth_qrf = max_depth_qrf
        self.random_state = random_state

        self.xgb_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
        }

        self.qrf_params = {
            "n_estimators": n_estimators_qrf,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "criterion": criterion,
            "max_depth": max_depth_qrf,
        }

        self.model = XGBQRF_SimpleModel(
            quantiles=quantiles,
            xgb_params=self.xgb_params,
            qrf_params=self.qrf_params,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
