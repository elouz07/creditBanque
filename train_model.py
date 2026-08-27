"""Entraîne et sauvegarde le modèle de risque de crédit.

Exécuter une seule fois avant de lancer l'application Streamlit :
    python train_model.py
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR.parent / "data" / "loan_approval_dataset.csv"
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "credit_model.pkl"
RESULTS_PATH = APP_DIR / "comparaison_modeles.csv"

FEATURES = [
    "no_of_dependents", "education_encoded", "self_employed_encoded",
    "income_annum", "loan_amount", "loan_term", "cibil_score",
    "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value", "loan_to_income_ratio",
    "total_assets_value", "loan_to_assets_ratio",
]


def prepare_data(path: Path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    for column in df.select_dtypes(include="object"):
        df[column] = df[column].str.strip()

    df = df.drop_duplicates().copy()

    # Une valeur d'actif négative est invalide : elle est remplacée par zéro.
    asset_columns = [
        "residential_assets_value", "commercial_assets_value",
        "luxury_assets_value", "bank_asset_value",
    ]
    for column in asset_columns:
        df[column] = df[column].clip(lower=0)

    # Bornage IQR des montants financiers pour limiter les valeurs extrêmes.
    financial_columns = ["income_annum", "loan_amount", *asset_columns]
    for column in financial_columns:
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[column] = df[column].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    df["education_encoded"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
    df["self_employed_encoded"] = df["self_employed"].map({"Yes": 1, "No": 0})
    df["default_flag"] = df["loan_status"].map({"Rejected": 1, "Approved": 0})

    df["loan_to_income_ratio"] = df["loan_amount"] / df["income_annum"].replace(0, 1)
    df["total_assets_value"] = df[asset_columns].sum(axis=1)
    df["loan_to_assets_ratio"] = df["loan_amount"] / df["total_assets_value"].replace(0, 1)

    X = df[FEATURES].fillna(0)
    y = df["default_flag"]
    return X, y


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {DATA_PATH}")

    X, y = prepare_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # La cible est suffisamment équilibrée (~62 % / ~38 %) : SMOTE n'est pas requis ici.
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    results = pd.DataFrame([{
        "model": "Random Forest optimisée",
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction),
        "recall": recall_score(y_test, prediction),
        "f1_score": f1_score(y_test, prediction),
        "roc_auc": roc_auc_score(y_test, probability),
    }])

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    results.to_csv(RESULTS_PATH, index=False)

    print("Meilleurs paramètres :", search.best_params_)
    print(results.round(3).to_string(index=False))
    print(f"Modèle sauvegardé : {MODEL_PATH}")


if __name__ == "__main__":
    main()
