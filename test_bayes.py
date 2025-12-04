import unittest
import numpy as np
from custom_naive_bayes import CustomGaussianNB
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TestCustomGaussianNB(unittest.TestCase):

    def test_simple_case(self):
        # wlasne dane test
        X = np.array([[-1], [-2], [-3], [1], [2], [3]])
        y = np.array([0, 0, 0, 1, 1, 1])

        model = CustomGaussianNB()
        model.fit(X, y)
        predictions = model.predict(X)

        self.assertTrue(np.array_equal(y, predictions))

    def test_wine_dataset(self):

        X, y = load_wine(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = CustomGaussianNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        print(f"\nTest Wine dokładność: {acc:.4f}")

        self.assertGreater(acc, 0.90)

    def test_breast_cancer_dataset(self):

        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = CustomGaussianNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        print(f"\nTestowanie Breast Cancer dokładność: {acc:.4f}")
        self.assertGreater(acc, 0.90)


if __name__ == '__main__':
    unittest.main()