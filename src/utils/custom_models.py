import pandas as pd
from quantile_forest import RandomForestQuantileRegressor


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
