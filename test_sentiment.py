import unittest
import numpy as np
from custom_naive_bayes import CustomGaussianNB
from text_helper import TextToNumConverter


class TestTweetSentiment(unittest.TestCase):

    def setUp(self):
        # Przykładowy dataset
        self.train_tweets = [
            "I love this product, it is amazing",
            "Best day ever, so happy and joyful",
            "Fantastic job, well done",
            "I hate this, it is terrible and bad",
            "Worst experience ever, sad and angry",
            "Do not buy this, it is garbage waste"
        ]
        # Etykiety: 1 = Pozytywny, 0 = Negatywny
        self.train_labels = np.array([1, 1, 1, 0, 0, 0])

        self.converter = TextToNumConverter()

        self.X_train = self.converter.fit_transform(self.train_tweets)

        self.model = CustomGaussianNB()
        self.model.fit(self.X_train, self.train_labels)

    def test_positive_tweet(self):
        """ Sprawdza czy model wykryje pozytywny wydźwięk. """
        new_tweet = ["What an amazing and happy day"]

        # Zmieniamy tekst na liczby
        X_new = self.converter.transform(new_tweet)
        prediction = self.model.predict(X_new)

        print(f"\nTweet: '{new_tweet[0]}'")
        print(f"Predykcja: {'Pozytywny' if prediction[0] == 1 else 'Negatywny'}")

        self.assertEqual(prediction[0], 1)

    def test_negative_tweet(self):
        """ Sprawdza czy model wykryje negatywny wydźwięk. """
        new_tweet = ["Terrible waste of time, angry"]

        # Konwersja na liczby
        X_new = self.converter.transform(new_tweet)
        prediction = self.model.predict(X_new)

        print(f"\nTweet: '{new_tweet[0]}'")
        print(f"Predykcja: {'Pozytywny' if prediction[0] == 1 else 'Negatywny'}")

        self.assertEqual(prediction[0], 0)

    def test_mixed_vocabulary(self):
        'Slowa nie znane przez model są ignorowane'

        # Słowo 'blabla' nie istnieje w treningu, ale 'hate' i 'terrible' tak.
        new_tweet = ["blabla terrible blabla hate"]

        X_new = self.converter.transform(new_tweet)
        prediction = self.model.predict(X_new)

        self.assertEqual(prediction[0], 0, "Wyłapuje negatywne słowa mimo szumu")


if __name__ == '__main__':
    unittest.main()