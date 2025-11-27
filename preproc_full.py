#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def main(train_path="data/train.csv", test_path="data/test.csv"):
    print("Loading CSV files...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    TARGET = "retention_status"
    ID_COL = "founder_id"

    if TARGET not in train.columns:
        raise ValueError(f"Train CSV missing target column '{TARGET}'")

    # Extract test IDs
    if ID_COL in test.columns:
        test_ids = test[ID_COL].copy()
    else:
        test_ids = pd.Series(test.index)

    # Remove ID column from train & test
    if ID_COL in train.columns:
        train = train.drop(columns=[ID_COL])
    if ID_COL in test.columns:
        test = test.drop(columns=[ID_COL])

    # Separate y
    y_train = train[TARGET].copy()
    train = train.drop(columns=[TARGET])

    # Identify numeric & categorical
    num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()

    # Impute numeric (median)
    num_impute = {col: train[col].median() for col in num_cols}
    for col, val in num_impute.items():
        train[col] = train[col].fillna(val)
        if col in test.columns:
            test[col] = test[col].fillna(val)

    # Impute categorical (mode)
    cat_impute = {}
    for col in cat_cols:
        mode = train[col].mode()
        cat_impute[col] = mode.iloc[0] if not mode.empty else "Missing"
        train[col] = train[col].fillna(cat_impute[col])
        if col in test.columns:
            test[col] = test[col].fillna(cat_impute[col])

    # One-hot encode
    combined = pd.concat(
        [train.assign(_is_train=1), test.assign(_is_train=0)],
        ignore_index=True
    )

    combined = pd.get_dummies(combined, drop_first=True)

    # Split back
    train_processed = combined[combined["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
    test_processed = combined[combined["_is_train"] == 0].drop(columns=["_is_train"]).reset_index(drop=True)

    # Scale numeric
    numeric_final = train_processed.select_dtypes(include=['int64','float64']).columns.tolist()
    scaler = StandardScaler()
    train_processed[numeric_final] = scaler.fit_transform(train_processed[numeric_final])
    test_processed[numeric_final] = scaler.transform(test_processed[numeric_final])

    # Save files
    train_processed.to_csv("X_train_processed.csv", index=False)
    test_processed.to_csv("X_test_processed.csv", index=False)
    y_train.to_csv("y_train_processed.csv", index=False)
    test_ids.to_csv("test_ids.csv", index=False)

    print("Saved:")
    print(" - X_train_processed.csv")
    print(" - X_test_processed.csv")
    print(" - y_train_processed.csv")
    print(" - test_ids.csv")
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
