import numpy as np
import re


class TextToNumConverter:
    def __init__(self):
        self.vocabulary = []  # Lista unikalnych słów
        self.vocab_map = {}  # Słownik {słowo: indeks}

    def fit_transform(self, text_list):
        """
        1. Uczy się słownika z podanych tekstów.
        2. Zwraca macierz liczbową (numpy array).
        """
        # Krok 1: Buduje słownik (unikalne słowa)
        unique_words = set()
        clean_texts = [self._clean(t) for t in text_list]

        for tokens in clean_texts:
            unique_words.update(tokens)

        self.vocabulary = sorted(list(unique_words))
        self.vocab_map = {word: i for i, word in enumerate(self.vocabulary)}

        # Krok 2: Zamienia na liczby
        return self._to_matrix(clean_texts)

    def transform(self, text_list):
        """
        Zamienia nowe teksty na macierz, używając już nauczonego słownika.
        """
        clean_texts = [self._clean(t) for t in text_list]
        return self._to_matrix(clean_texts)

    def _to_matrix(self, clean_texts):
        """
        Tworzy macierz (Liczba Próbek x Liczba Słów w Słowniku).
        """
        n_samples = len(clean_texts)
        n_features = len(self.vocabulary)

        # Tworzymy macierz zer
        matrix = np.zeros((n_samples, n_features), dtype=np.float64)

        for row_idx, tokens in enumerate(clean_texts):
            for token in tokens:
                if token in self.vocab_map:
                    col_idx = self.vocab_map[token]
                    # Wstawiamy 1 jeśli słowo występuje (można też += 1 żeby zliczać)
                    matrix[row_idx, col_idx] += 1.0

        return matrix

    def _clean(self, text):
        """
        Proste czyszczenie: małe litery i usuwanie znaków specjalnych.
        """
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()