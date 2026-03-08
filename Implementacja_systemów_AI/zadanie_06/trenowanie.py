import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import bentoml

# Załaduj zbiór danych
data = pd.read_csv("titanic.csv")  # Użyj odpowiedniego linku lub metody do wczytania danych
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]  # Wybierz odpowiednie cechy
y = data['Survived']

# Przetwórz dane (zakładam proste przetwarzanie)
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X['Age'].fillna(X['Age'].median(), inplace=True)
X['Fare'].fillna(X['Fare'].median(), inplace=True)

# Podziel dane
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Wytrenuj RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Wytrenuj LogisticRegression
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X_train, y_train)

# Zapisz modele do BentoML
bentoml.sklearn.save_model("titanic_rf", rf_model)
bentoml.sklearn.save_model("titanic_logreg", logreg_model)
