import pandas as pd
import numpy as np
import itertools
from sklearn.neural_network import MLPClassifier
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


    print("\nSampling 40% of full data for train + validation...")

    X_40, _, y_40, _ = train_test_split(
        X, y,
        test_size=0.6,        
        random_state=42,   
        stratify=y
    )

    print("  Sampled X_40:", X_40.shape)
    print("  Sampled y_40:", y_40.shape)


    print("\nCreating 20%-20% train-validation split from sampled data...")

    X_train, X_val, y_train, y_val = train_test_split(
        X_40, y_40,
        test_size=0.5,       # 50% of 40% = 20%
        random_state=42,
        stratify=y_40
    )

    print("  Train (20%):", X_train.shape)
    print("  Val   (20%):", X_val.shape)

    X_train = ensure_numeric(X_train)
    X_val = ensure_numeric(X_val)
    X_test = ensure_numeric(X_test)

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)


    print("\nStarting Neural Network grid search...")

    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (64,64), (50, 64)],
        "alpha": [0.0001, 0.001, 0.0005],
        "learning_rate_init": [0.001, 0.01, 0.02, 0.03]
    }

    combos = list(itertools.product(
        param_grid["hidden_layer_sizes"],
        param_grid["alpha"],
        param_grid["learning_rate_init"]
    ))

    best_acc = -1
    best_params = None

    print(f"Total combinations: {len(combos)}\n")

    for i, (hls, alpha, lr) in enumerate(combos, start=1):
        print(f"Training model {i}/{len(combos)} → layers={hls}, alpha={alpha}, lr={lr}")

        model = MLPClassifier(
            hidden_layer_sizes=hls,
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=500,
            random_state=0,
        )

        try:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_val_s)
            acc = accuracy_score(y_val, preds)
            print(f"→ Val Accuracy = {acc:.4f}\n")

            if acc > best_acc:
                best_acc = acc
                best_params = (hls, alpha, lr)

        except Exception as e:
            print("Model failed:", e)


    print("=" * 70)
    print("BEST MODEL FOUND")
    print("=" * 70)
    print(f"Best params: hidden_layers={best_params[0]}, alpha={best_params[1]}, lr={best_params[2]}")
    print(f"Best Validation Accuracy: {best_acc:.4f}\n")


    print("Training best model on SAME 20% training data...")

    final_model = MLPClassifier(
        hidden_layer_sizes=best_params[0],
        alpha=best_params[1],
        learning_rate_init=best_params[2],
        max_iter=1000,
        random_state=0,
    )

    final_model.fit(X_train_s, y_train)

    print("\nGenerating predictions on test set...")
    test_preds = final_model.predict(X_test_s)

    submission = pd.DataFrame({
        "founder_id": test_ids,
        "retention_status": test_preds
    })

    submission.to_csv("submission_nn_20_20.csv", index=False)
    print("Saved submission_nn_20_20.csv")

    print("\nPrediction distribution:")
    print(submission["retention_status"].value_counts())


if __name__ == "__main__":
    main()
