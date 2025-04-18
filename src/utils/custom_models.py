import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.base import BaseEstimator, TransformerMixin


class VotingRandomForestQuantileRegressor:
    def __init__(
        self,
        model_variants=[
            "full_model_remove_station_identication",
            "france_remove_station_identication",
            "brazil_remove_station_identication",
        ],
    ):
        self.models = {}
        self.variants = model_variants
        self.adjusted_weights = {}
        self.adjust_weights()

    def adjust_weights(self, location_confidence=1.4):
        self.adjusted_weights = {
            0: {
                k: (
                    location_confidence
                    if "brazil" in k
                    else 1 if "full" in k else (2 - location_confidence)
                )
                for k in self.variants
            },
            1: {
                k: (
                    location_confidence
                    if "france" in k
                    else 1 if "full" in k else (2 - location_confidence)
                )
                for k in self.variants
            },
        }

    @staticmethod
    def adjust_dataset(variant_name, X, y=None, predict=False):
        X_train = X.copy()
        if isinstance(y, pd.Series):
            X_train["y"] = y
        if "remove_station_identication" in variant_name:
            X_train = X_train.drop(columns=["station_code"])
            X_train = X_train.drop(columns=["catchment"])
            X_train = X_train.drop(columns=["latitude"])
            X_train = X_train.drop(columns=["longitude"])

        if "france" in variant_name and not predict:
            X_train = X_train[X_train["north_hemisphere"] == 1]

        if "brazil" in variant_name and not predict:
            X_train = X_train[X_train["north_hemisphere"] == 0]

        if isinstance(y, pd.Series):
            return X_train.drop(columns=["y"]), X_train["y"]
        else:
            return X_train

    def fit(self, X, y):
        """Train models based on given data."""

        for variant in self.variants:
            print(f"Training for variant {variant}")
            self.models[variant] = RandomForestQuantileRegressor(
                n_estimators=60,
                min_samples_split=5,
                min_samples_leaf=15,
                max_features=None,
                max_depth=20,
                bootstrap=True,
            )
            X_adj, y_adj = self.adjust_dataset(variant, X, y)
            print(f"Training on {X_adj.shape[0]} samples")
            self.models[variant].fit(X_adj, y_adj)

    def predict(self, X, weighted=True, selected_variants=None):
        """Predict based on given data."""

        if selected_variants is None:
            selected_variants = self.variants

        variant_predictions = pd.DataFrame()
        total_predictions = pd.DataFrame()

        for variant in selected_variants:
            print(f"Predicting for {variant}")
            X_adj = self.adjust_dataset(variant, X, predict=True)

            variant_predictions.iloc[0:0]
            variant_predictions["north_hemsiphere"] = X_adj["north_hemisphere"]

            variant_predictions["mean"] = self.models[variant].predict(
                X_adj,
                quantiles="mean",
                aggregate_leaves_first=False,
            )
            quantiles = self.models[variant].predict(X_adj, quantiles=[0.05, 0.95])

            variant_predictions["lower"] = quantiles[:, 0]
            variant_predictions["upper"] = quantiles[:, 1]

            numerical_cols = ["mean", "lower", "upper"]

            if weighted:
                for key in self.adjusted_weights.keys():
                    variant_predictions.loc[
                        variant_predictions["north_hemsiphere"] == key, numerical_cols
                    ] *= self.adjusted_weights[key][variant]

            if total_predictions.empty:
                total_predictions[numerical_cols] = variant_predictions[numerical_cols]
            else:
                total_predictions[numerical_cols] += variant_predictions[numerical_cols]

        total_predictions[numerical_cols] /= len(self.variants)
        return total_predictions


class SnowIndexComputeTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        altitude_weight=1,
        temp_weight=1.2,
        precip_weight=0.2,
        temp_col_name="temperatures",
        rain_col_name="precipitations",
    ):
        self.altitude_weight = altitude_weight
        self.temp_weight = temp_weight
        self.precip_weight = precip_weight
        self.temp_col_name = temp_col_name
        self.rain_col_name = rain_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        required_columns = ["altitude", self.temp_col_name, self.rain_col_name]
        if not all(col in X_transformed.columns for col in required_columns):
            raise ValueError(
                f"DataFrame must contain the following columns: {required_columns}"
            )

        # Make all temps positive to get relevant increasing output
        min_temp = X_transformed[self.temp_col_name].min()
        if min_temp <= 0:
            X_transformed[self.temp_col_name] += abs(min_temp) + 1

        # Compute index
        snow_index = (
            (
                X_transformed["altitude"] * self.altitude_weight
            )  # More height = more snow
            * (
                X_transformed[self.rain_col_name] * self.precip_weight
            )  # More rain = more snow
            / (
                X_transformed[self.temp_col_name] * self.temp_weight
            )  # Lower temps = more snow
        )

        # handle infinites et NaN
        snow_index.replace([np.inf, -np.inf], np.nan, inplace=True)
        snow_index.fillna(0, inplace=True)

        # Normalize
        snow_index = (snow_index - snow_index.min()) / (
            snow_index.max() - snow_index.min()
        )

        X_transformed["snow_index"] = snow_index

        return X_transformed


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
            preds = [None] * len(X)

            for clust_index in range(self.number_of_clusters):
                print(f"Predicting cluster {clust_index}")
                X_pred = X[X[self.specialized_col] == clust_index].copy(deep=True)

                cluster_preds = self.models[i][clust_index].predict(
                    X_pred[self.qrf_features[clust_index]], quantiles=quantiles
                )

                # Repositionne les prédictions au bon endroit
                for idx, pred in zip(X_pred.index, cluster_preds):
                    preds[idx] = pred

            predictions[i] = preds  # Liste alignée avec X.index

        return predictions
