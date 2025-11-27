#!/usr/bin/env python3
import pandas as pd
import numpy as np
import itertools
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def ensure_numeric(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c])
            except:
                df[c] = df[c].astype("category").cat.codes
    return df


def main():
    print("Loading processed CSV files...")

    X = pd.read_csv("X_train_processed.csv")
    y = pd.read_csv("y_train_processed.csv").iloc[:, 0].values.ravel()
    X_test = pd.read_csv("X_test_processed.csv")
    test_ids = pd.read_csv("test_ids.csv").iloc[:, 0].values.ravel()

    print(f"Loaded X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}")

    # ------------------------------------------------------------
    # ✅ FIX: CONVERT STRING LABELS → NUMERIC (REQUIRED FOR XGBOOST)
    # ------------------------------------------------------------
    y = pd.Series(y).map({"Left": 0, "Stayed": 1}).astype(int)

    # Ensure numeric features
    X = ensure_numeric(X)
    X_test = ensure_numeric(X_test)

    # ------------------------------------------------------------
    # ✅ 80–20 TRAIN / VALIDATION SPLIT
    # ------------------------------------------------------------
    print("\nCreating 80-20 split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ------------------------------------------------------------
    # ✅ SCALING (OPTIONAL FOR XGBOOST BUT KEPT FOR CONSISTENCY)
    # ------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # ------------------------------------------------------------
    # ✅ GRID SEARCH
    # ------------------------------------------------------------
    param_grid = {
        "n_estimators": [200, 400, 100, 500],
        "max_depth": [3, 5, 7, 100],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    combos = list(itertools.product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["subsample"],
        param_grid["colsample_bytree"]
    ))

    best_acc = -1
    best_params = None

    print("\nGrid search starting...\n")

    for i, (n_est, depth, lr, subs, cols) in enumerate(combos, start=1):
        print(
            f"Model {i}/{len(combos)} → "
            f"n_estimators={n_est}, max_depth={depth}, "
            f"lr={lr}, subsample={subs}, colsample={cols}"
        )

        model = XGBClassifier(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            subsample=subs,
            colsample_bytree=cols,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)

        acc = accuracy_score(y_val, preds)
        print(f"  Val Accuracy = {acc:.4f}\n")

        if acc > best_acc:
            best_acc = acc
            best_params = {
                "n_estimators": n_est,
                "max_depth": depth,
                "learning_rate": lr,
                "subsample": subs,
                "colsample_bytree": cols
            }

    # ------------------------------------------------------------
    # ✅ BEST MODEL SUMMARY
    # ------------------------------------------------------------
    print("=" * 60)
    print("Best XGBoost Model:")
    print(best_params, " Val Accuracy:", best_acc)
    print("=" * 60)

    # ------------------------------------------------------------
    # ✅ RETRAIN ON FULL DATASET
    # ------------------------------------------------------------
    print("\nTraining best XGBoost on FULL dataset...")

    X_full_scaled = scaler.fit_transform(X)

    final_model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X_full_scaled, y)

    # ------------------------------------------------------------
    # ✅ FINAL PREDICTION
    # ------------------------------------------------------------
    print("Predicting on test set...")
    final_preds_num = final_model.predict(X_test_scaled)

    # ------------------------------------------------------------
    # ✅ OPTIONAL: CONVERT 0/1 BACK TO ORIGINAL LABELS
    # ------------------------------------------------------------
    label_map = {0: "Left", 1: "Stayed"}
    final_preds = pd.Series(final_preds_num).map(label_map).values

    submission = pd.DataFrame({
        "founder_id": test_ids,
        "retention_status": final_preds
    })

    submission.to_csv("submission_xgboost.csv", index=False)
    print("Saved submission_xgboost.csv")

    print("\nPrediction distribution:")
    print(submission["retention_status"].value_counts())


if __name__ == "__main__":
    main()
