import os
import matplotlib.pyplot as plt


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
