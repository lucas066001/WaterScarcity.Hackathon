from typing import Callable, List, Any


def create_predict_function(
        model_list: List[Any],
        i: int,
        model: str
        ) -> Callable:
    """
    Creates a prediction function based on the specified model type.

    Args:
        model_list (List[Any]): A list of trained models.
        i (int): The index of the model to use from the list.
        model (str): The type of model, either 'mapie' or other types.

    Returns:
        Callable: A function that takes input data X and returns predictions.
    """
    def predict(X):
        if model == "mapie":
            return model_list[i].predict(X)[0]
        else:
            return model_list[i].predict(X)
    return predict


def create_quantile_function(
        models: List[Any],
        i: int,
        model: str,
        alpha: float = .1
        ) -> Callable:
    """
    Creates a quantile prediction function based on the specified model type.

    Args:
        model_list (List[Any]): A list of trained models.
        i (int): The index of the model to use from the list.
        model (str): The type of model, either 'mapie' or 'qrf'.
        alpha (float): The confidence level for the quantile prediction.

    Returns:
        Callable: A function that takes input data X 
        and returns quantile predictions.
    """
    def predict_quantile(X):
        if model == "mapie":
            return models[i].predict(X)[1]
        elif model == "qrf":
            return models[i].predict(X, quantiles=[alpha / 2, 1 - alpha / 2])
        raise ValueError(f"Unsupported model type: {model}")
    return predict_quantile
