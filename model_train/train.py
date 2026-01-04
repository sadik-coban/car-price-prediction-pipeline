import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split


# pvDf = pd.read_csv("data_bmw_exported.csv")
def train(df):
    X = df.drop(
        columns=["Fiyat", "Segment_Encoded"],
        axis=1,
    )

    y = df["Fiyat"]

    cat_cols = X.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        X[col] = X[col].fillna("Bilinmiyor").astype(str)

    for col in cat_cols:
        X[col] = X[col].astype("category")

    cat_features_indices = [
        i for i, col in enumerate(X.columns) if X[col].dtype == "category"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.01,
        depth=8,
        loss_function="RMSE",
        verbose=100,
        task_type="CPU",
    )

    model.fit(
        X_train, y_train, cat_features=cat_features_indices, eval_set=(X_test, y_test)
    )

    model.save_model("catboost_model.cbm")
