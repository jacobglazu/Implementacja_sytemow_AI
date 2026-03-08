import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
import bentoml

# Załaduj zbiór danych
def load_and_preprocess_titanic():
    data = fetch_openml(data_id=40945, as_frame=True)
    df = data.frame
    df = df.drop(columns=["name", "ticket", "cabin", "body", "boat", "home.dest"])
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)
    X = df.drop(columns=["survived"])
    y = df["survived"].astype(int)
    return X, y

# Podziel dane
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def main():
    X, y = load_and_preprocess_titanic()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Wytrenuj RandomForestClassifier
    titanic_RF = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    titanic_RF.fit(X_train, y_train)
    accuracy = titanic_RF.score(X_test, y_test)

    # Wytrenuj LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    titanic_logreg = LogisticRegression(max_iter=1000, solver="lbfgs")
    titanic_logreg.fit(X_train_scaled, y_train)

    # Zapisz modele do BentoML (jeden model na raz)
    rf_tag = bentoml.sklearn.save_model(
        "titanic_rf_classifier",
        titanic_RF,
        signatures={
            "predict": {"batchable": True, "batch_dim": 0},
            "predict_proba": {"batchable": True, "batch_dim": 0},
        },
        metadata={"accuracy": accuracy},
    )
    logreg_tag = bentoml.sklearn.save_model(
        "titanic_logreg_classifier",
        titanic_logreg,
        signatures={
            "predict": {"batchable": True, "batch_dim": 0},
            "predict_proba": {"batchable": True, "batch_dim": 0},
        },
        metadata={"accuracy": accuracy},
    )
    print(f"RandomForest model saved: {rf_tag}")
    print(f"LogisticRegression model saved: {logreg_tag}")

if __name__ == "__main__":
    main()
