from typing import Callable, List, Any
import numpy as np


def create_predict_function(model_list: List[Any], i: int, model: str) -> Callable:
    """
    Creates a prediction function based on the specified model type.

    Parameters:
        model_list (List[Any]): A list of trained models.
        i (int): The index of the model to use from the list.
        model (str): The type of model, either 'mapie' or other types.

    Returns:
        Callable: A function that takes input data X and returns predictions.
    """

    def predict(X):
        if model == "mapie":
            return model_list[i].predict(X)[0]
        elif model == "lgbm":
            return model_list[i][1].predict(X)
        elif model == "xgb":
            return model_list[i][0.5].predict(X)
        elif model == "qrf_bagging":
            return model_list[i][1].predict(X, quantiles="mean")
        elif model == "gbr":
            return model_list[i]["median"].predict(X)
        elif model == "deep_ensemble":
            y_pred_deep = []
            for m in model_list[i]:
                y_pred_deep.append(m.predict(X))
            y_pred_deep = np.array(y_pred_deep)
            return np.mean(y_pred_deep, axis=0)
        elif model == "xgb_qrf" or model == "xgb_qrf_simple":
            return model_list[i].predict(X)[:, 1]
        else:
            return model_list[i].predict(X)

    return predict


def create_quantile_function(
    models: List[Any], i: int, model: str, alpha: float = 0.1
) -> Callable:
    """
    Creates a quantile prediction function based on the specified model type.

    Parameters:
        model_list (List[Any]): A list of trained models.
        i (int): The index of the model to use from the list.
        model (str): The type of model, either 'mapie' or 'qrf'.
        alpha (float): The confidence level for the quantile prediction.

    Returns:
        Callable: A function that takes input data X
        and returns quantile predictions.
    """

    def predict_quantile(X):
        print(f"model : {model}")
        if model == "mapie":
            return models[i].predict(X)[1]
        if model == "lgbm":
            return np.stack([models[i][0].predict(X), models[i][2].predict(X)], axis=1)
        if model == "xgb":
            quantiles = [alpha / 2, 1 - alpha / 2]
            return np.stack(
                [models[i][q].predict(X) for q in quantiles],
                axis=1,
            )
        elif model == "qrf":
            return models[i].predict(X, quantiles=[alpha / 2, 1 - alpha / 2])
        elif model == "gbr":
            return np.stack(
                [models[i]["lower"].predict(X), models[i]["upper"].predict(X)]
            )
        elif model == "xgb_qrf" or model == "xgb_qrf_simple":
            predictions = models[i].predict(X)
            return np.stack([predictions[:, 0], predictions[:, 2]], axis=1)
        elif model == "qrf_bagging":
            return np.stack(
                [
                    [
                        est.predict(X, quantiles=[alpha / 2, 1 - alpha / 2])
                        for est in models[i].estimators_
                    ]
                ],
                axis=1,
            )
        elif model == "deep_ensemble":
            y_pred_deep = []
            for m in models[i]:
                y_pred_deep.append(m.predict(X))
            y_pred_deep = np.array(y_pred_deep)
            return np.quantile(y_pred_deep, [alpha / 2, 1 - alpha / 2], axis=0)
        raise ValueError(f"Unsupported model type: {model}")

    return predict_quantile
