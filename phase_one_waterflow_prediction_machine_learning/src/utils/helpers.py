import os
import matplotlib.pyplot as plt
import numpy as np


def save_or_create(plt: plt.Figure, save_path: str):
    """
    Save a plot to a file, creating the directory if it does not exist.
    Parameters:
        plt: The plot to save.
        save_path (str): The full file path where the plot will be saved.

    Returns:
        None
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)


def add_lagged_features(df, features, lags, is_target=False):
    """
    Adds lagged versions of the given features to the dataframe.
    - features: list of feature column names
    - lags: number of lags to add (int or dict)
    - is_target: if True, sort by ObsDate in descending order
    """
    df = df.copy()
    if isinstance(lags, int):
        lags = {feature: lags for feature in features}

    suffix = "_lag_w" if not is_target else "_w"

    for feature in features:
        for lag in range(1, lags[feature] + 1):
            df[f"{feature}{suffix}{lag}"] = df[feature].shift(lag).fillna(-1)
    return df


def add_lagged_features_per_station(df, features, lags, is_target=False):
    """
    Applies lagged feature generation per station.
    """
    return df.groupby("station_code", group_keys=False).apply(
        lambda group: add_lagged_features(
            group.sort_values("ObsDate", ascending=not is_target),
            features,
            lags,
            is_target,
        )
    )


def check_columns_exist(df, columns):
    """
    Check if the specified columns exist in the DataFrame.
    """
    missing_columns = [col for col in columns if col not in df.columns]
    return missing_columns


def custom_nll_scorer(y_true, y_pred):
    if len(y_pred) == 0:
        raise ValueError("y_pred is empty")

    y_lower, y_pred, y_upper = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    # Estimate sigma using the 95% CI approximation
    sigma = (y_upper - y_lower) / 3.29

    # Gaussian-like NLL (your formula)
    nll = np.mean(np.log(sigma) + np.abs(y_true - y_pred) / (2 * np.abs(sigma)))

    # Optionally: add coverage or interval size as penalty terms
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    interval_size = np.mean(y_upper - y_lower)

    # Example: penalize low coverage and large intervals (you can adjust weights)
    total_loss = nll + 2 * (0.9 - coverage) ** 2 + 0.1 * interval_size

    return total_loss


def get_y_train(df, number_of_weeks=4):
    y_train = {}
    df_copy = df.copy()
    if "water_flow_week1" not in df_copy.columns:
        raise ValueError("water_flow_week1 not in columns")
    for i in range(0, number_of_weeks):
        y_train[i] = df[f"water_flow_week{i+1}"]
        df_copy.drop(columns=[f"water_flow_week{i+1}"], inplace=True)
    return df_copy, y_train
