from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import download
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from torch import FloatTensor

from neucube.encoder import Probability
from experiments.params import stop_words, random_seed


class TextPrep:

    def __init__(self, svd_components: int = 1000, prob_iterations: int = 500):
        download('punkt')
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer(
            strip_accents="ascii",
            lowercase=True,
            preprocessor=self.preprocess_and_stem, stop_words=stop_words
            )
        self.svd = TruncatedSVD(n_components=svd_components, random_state=random_seed)
        self.encoder = Probability(iterations=prob_iterations)

    def preprocess_and_stem(self, text: str):
        text = text.lower()
        tokens = word_tokenize(text)
        stems = [self.stemmer.stem(token) for token in tokens]
        preprocessed_text = ' '.join(stems)
        return preprocessed_text

    def preprocess_dataset(self, sklearn_dataset, lsa=True, spikes=True) -> tuple[FloatTensor, FloatTensor]:
        train_x = self.vectorizer.fit_transform(sklearn_dataset.data)
        if lsa:
            train_x = self.svd.fit_transform(train_x)
        train_x = FloatTensor(train_x).relu()  # no negative values
        if spikes:
            train_x = self.encoder.encode_dataset(train_x)

        train_y = sklearn_dataset.target
        train_y = FloatTensor(train_y)
        return train_x, train_y
