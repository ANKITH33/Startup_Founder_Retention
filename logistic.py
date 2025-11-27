import pandas as pd
import numpy as np
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(2)

def load_processed():
    print("Loading processed datasets...")

    X = pd.read_csv("X_train_processed.csv")
    y = pd.read_csv("y_train_processed.csv")
    X_test = pd.read_csv("X_test_processed.csv")
    test_ids = pd.read_csv("test_ids.csv")

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if isinstance(test_ids, pd.DataFrame):
        test_ids = test_ids.iloc[:, 0]

    print(f"  X_train: {X.shape}")
    print(f"  y_train: {y.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  test_ids: {test_ids.shape}")

    return X, y, X_test, test_ids


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

    X, y, X_test, test_ids = load_processed()

    if "founder_id" in X.columns:
        X = X.drop(columns=["founder_id"])

    print("\nCreating train/validation split...")
    strat = y if y.nunique() > 1 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=strat
    )

    print("  Train:", X_train.shape)
    print("  Val:  ", X_val.shape)

    X_train = ensure_numeric(X_train)
    X_val = ensure_numeric(X_val)
    X_test = ensure_numeric(X_test)

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("\nStarting Logistic Regression grid search...")

    param_grid = {
        "C": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "class_weight": [None, "balanced"]
    }

    combos = list(itertools.product(
        param_grid["C"],
        param_grid["penalty"],
        param_grid["solver"],
        param_grid["class_weight"]
    ))

    best_acc = -1
    best_params = None
    best_model = None

    print(f"Total combinations: {len(combos)}\n")

    for i, (C, penalty, solver, cw) in enumerate(combos, start=1):

        print(
            f"Training model {i}/{len(combos)} → "
            f"C={C}, penalty={penalty}, solver={solver}, class_weight={cw}"
        )

        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            class_weight=cw,
            max_iter=2000,
        )

        try:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_val_s)
            acc = accuracy_score(y_val, preds)

            print(f"→ Val Accuracy = {acc:.4f}\n")

            if acc > best_acc:
                best_acc = acc
                best_params = (C, penalty, solver, cw)
                best_model = model

        except Exception as e:
            print("Model failed:", e)

    print("=" * 70)
    print("BEST MODEL FOUND")
    print("=" * 70)
    print(
        f"Best params: C={best_params[0]}, "
        f"penalty={best_params[1]}, "
        f"solver={best_params[2]}, "
        f"class_weight={best_params[3]}"
    )
    print(f"Best Validation Accuracy: {best_acc:.4f}\n")

    print("Retraining best Logistic Regression model on FULL training data...")

    X_full = ensure_numeric(X.copy())
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X_full)
    X_test_s_full = scaler_full.transform(ensure_numeric(X_test))

    final_model = LogisticRegression(
        C=best_params[0],
        penalty=best_params[1],
        solver=best_params[2],
        class_weight=best_params[3],
        max_iter=2000,
    )

    final_model.fit(X_full_s, y)

    print("\nGenerating predictions on test set...")
    test_preds = final_model.predict(X_test_s_full)

    submission = pd.DataFrame({
        "founder_id": test_ids,
        "retention_status": test_preds
    })

    submission.to_csv("submission_logistic.csv", index=False)
    print("Saved submission_logistic.csv")

    print("\nPrediction distribution:")
    print(submission["retention_status"].value_counts())


if __name__ == "__main__":
    main()
