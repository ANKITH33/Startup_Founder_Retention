#!/usr/bin/env python3
import pandas as pd
import numpy as np
import itertools
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(2)

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
    y = pd.read_csv("y_train_processed.csv").iloc[:, 0]
    X_test = pd.read_csv("X_test_processed.csv")
    test_ids = pd.read_csv("test_ids.csv").iloc[:, 0]

    print(f"Loaded X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}")


    X = ensure_numeric(X)
    X_test = ensure_numeric(X_test)

    
    print("\nSampling random 20% of training data...")
    X_20, _, y_20, _ = train_test_split(
        X,
        y,
        test_size=0.8,      
        random_state=None,  
        stratify=y
    )

    print(f"Sampled X: {X_20.shape}, y: {y_20.shape}")


    print("\nCreating train/validation split from 20% sample...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_20,
        y_20,
        test_size=0.20,       
        random_state=2,
        stratify=y_20
    )

    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_20_scaled = scaler.transform(X_20)        
    X_test_scaled = scaler.transform(X_test)   

    param_grid = {
        "C": [0.1, 1, 5, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }

    combos = list(itertools.product(
        param_grid["C"],
        param_grid["kernel"],
        param_grid["gamma"]
    ))

    best_acc = -1
    best_params = None
    best_model = None

    print("\nGrid search starting...\n")

    for i, (C, kernel, gamma) in enumerate(combos, start=1):
        print(f"Model {i}/{len(combos)}  →  C={C}, kernel={kernel}, gamma={gamma}")

        model = SVC(C=C, kernel=kernel, gamma=gamma)

        try:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
            acc = accuracy_score(y_val, preds)

            print(f"  Val Accuracy = {acc:.4f}\n")

            if acc > best_acc:
                best_acc = acc
                best_params = {"C": C, "kernel": kernel, "gamma": gamma}
                best_model = model

        except Exception as e:
            print("Model failed:", e)

    print("=" * 60)
    print("Best Model:")
    print(best_params, " Val Accuracy:", best_acc)
    print("=" * 60)

    print("\nTraining best SVM on SAME 20% dataset...")

    final_model = SVC(**best_params)
    final_model.fit(X_20_scaled, y_20)

    print("Predicting on test set...")
    final_preds = final_model.predict(X_test_scaled)

    submission = pd.DataFrame({
        "founder_id": test_ids,
        "retention_status": final_preds
    })

    submission.to_csv("submission_svm_20.csv", index=False)
    print("Saved submission_svm_20.csv")

    print("\nPrediction distribution:")
    print(submission["retention_status"].value_counts())


if __name__ == "__main__":
    main()
