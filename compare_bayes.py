import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from custom_naive_bayes import CustomGaussianNB


def compare_implementations(dataset_name, data_loader):

    print(f"\nZbiór danych: {dataset_name.upper()}")

    # ładowanie danych
    data = data_loader()
    X, y = data.data, data.target
    target_names = data.target_names

    # trenowanie
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Liczba próbek: {X.shape[0]}, Cechy: {X.shape[1]}, Klasy: {len(target_names)}")

    # Nasza implementacja
    custom_model = CustomGaussianNB()
    custom_model.fit(X_train, y_train)
    custom_preds = custom_model.predict(X_test)
    custom_acc = accuracy_score(y_test, custom_preds)

    # biblioteka Scikit-Learn
    sklearn_model = SklearnGaussianNB()
    sklearn_model.fit(X_train, y_train)
    sklearn_preds = sklearn_model.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_preds)

    # Wynik
    print(f"Nasza implementacja:   {custom_acc:.4f}")
    print(f"Scikit-Learn dokładność:        {sklearn_acc:.4f}")

    # sprawdzamy zgodność
    diff = custom_acc - sklearn_acc
    match_count = np.sum(custom_preds == sklearn_preds)
    total = len(y_test)

    if np.isclose(diff, 0):
        print(f"Wyniki są niemal identyczne (różnica {diff:.6f})")
    else:
        print(f"Różnica w dokładności wynosi {diff:.6f}")

    print(f"Zgodność predykcji co do próbki: {match_count}/{total} ({match_count / total * 100:.2f}%)")

# sprawdzenie danych z excela
datasets_to_check = [
    ("Wine", load_wine),
    ("Breast Cancer", load_breast_cancer)
]

for name, loader in datasets_to_check:
    compare_implementations(name, loader)