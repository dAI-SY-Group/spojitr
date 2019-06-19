"""
Spojitr text processing utils
"""

import functools
import logging
import nltk
import typing
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics.pairwise

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer


# *******************************************************************
# CONFIGURATION
# *******************************************************************


LOGGER = logging.getLogger()

TRANSLATION_TABLE = str.maketrans(dict.fromkeys(",!.?:;"))

CAMEL_CASE_REGEX = re.compile(r"([a-z])([A-Z])")
SNAKE_CASE_REGEX = re.compile(r"(.*?)_([a-zA-Z])")
STEMMER = SnowballStemmer("english")
STOPWORDS = set(stopwords.words("english"))


# *******************************************************************
# FUNCTIONS
# *******************************************************************


def download_nltk_data():
    """Fetch required nltk data

    :Note: this needs to be done only once, probably during installation
    """
    nltk.download("stopwords")
    nltk.download("word_tokenize")
    nltk.download("punkt")


def _split_combined_words(tokens: typing.Iterable) -> list:
    @functools.lru_cache(maxsize=1024)
    def do_split(token) -> list:
        splits = re.sub(CAMEL_CASE_REGEX, r"\1 \2", token).split()

        if len(splits) == 1:
            # not camel case -> try snake case
            return re.sub(SNAKE_CASE_REGEX, r"\1 \2", token).split()

        return splits

    res = []
    for token in tokens:
        res.extend(do_split(token))

    # LOGGER.info(do_split.cache_info())
    return res


# profiling showed, that stemming eats up a lot of time
# therefore we cache results
@functools.lru_cache(maxsize=1024)
def _stem(word):
    return STEMMER.stem(word)


def preprocess(text: str) -> str:
    """Apply text preprocessing chain, e.g. stop word removal, splitting, etc
    """
    trans_text = text.translate(TRANSLATION_TABLE)
    word_tokens = word_tokenize(trans_text)
    stop_words_removed = (word for word in word_tokens if word not in STOPWORDS)
    splitted_tokens = _split_combined_words(stop_words_removed)
    return " ".join([_stem(token) for token in splitted_tokens])


class CosineSimilarity:
    def __init__(self, docs: typing.Iterable):
        """Initialize with corpus of documents

        :param docs: list documents in form of tuples (doc_identifier, document_content)
        """
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 4))

        corpus = []
        # inverse index
        self._doc_id_2_idx = {}  # type: ignore
        for idx, (doc_id, doc_text) in enumerate(docs):
            self._doc_id_2_idx[doc_id] = idx
            corpus.append(doc_text)

        self._X = self._vectorizer.fit_transform(corpus)

    def get_similarities(self, queries: typing.Iterable):
        """Get all similarity values of trained corpus against queries

        cos_sim = CosineSimilarity( .... )
        sim = cos_sim.get_similarities([q0, q1, q2, ...])

        the similarities for q0 to [doc0, doc1, doc2, ....] would be

            sim_q0 = sim[:, 0].flatten()

        and for q2

            sim_q2 = sim[:, 2].flatten()

        :Returns: matrix of shape [number of corpus documents] x [number of queries]
                  i.e. each _column_ (in order of queries iterable) contains
                  the similarities values to all corpus documents
        """
        Y = self._vectorizer.transform(queries)

        # TODO: all value are already L2 normalized (TfidfVectorizer default),
        # therefore we also could use the more efficient pairwise.linear_kernel
        sim = sklearn.metrics.pairwise.cosine_similarity(self._X, Y)
        return sim

    def get_similarities_by_doc_id(self, queries: typing.Iterable, doc_ids: list):
        """Get similarity values for a _subset_ of all corpus documents

        cos_sim = CosineSimilarity( [ ("d0", content_0), ("d1", content_1), ...] )
        sim = cos_sim.get_similarities_by_doc_id([q0, q1, q2, ...], ["d0", "d4"] )

        the similarities for q0 to [d0, d4] would be

            sim_q0 = sim[:, 0].flatten()

        and for q2

            sim_q2 = sim[:, 2].flatten()

        :Returns: matrix of shape [len(doc_ids)] x [number of queries]
                  i.e. each _column_ (in order of queries iterable) contains
                  the similarities values to the _subset_ of corpus documents (in that order)
        """
        # calc all similarties
        sim = self.get_similarities(queries)

        # select only the requested documents (i.e. rows of similarities)
        # but for _all_ queries (i.e. columns)
        row_indices = [self._doc_id_2_idx[doc_id] for doc_id in doc_ids]
        return sim[row_indices, :]


def calculate_similarity(
    query_doc: str, corpus: typing.List[str]
) -> typing.List[float]:
    """Calculate similarity form 'query_doc' to documents in 'corpus'

    :param corpus: document corpus
    :Returns: list of similarity values in order of corpus documents
    """
    # 'fake' document ids, which are only technically required in this scenario
    doc_ids = [f"d{i}" for i in range(len(corpus))]
    calculator = CosineSimilarity(zip(doc_ids, corpus))

    matrix = calculator.get_similarities([query_doc])

    # select all values from first column, since we only have one query
    return matrix[:, 0].flatten()


# *******************************************************************
# TEST
# *******************************************************************


def test_preprocess():
    texts = [
        "This is an example text",
        "java method doThisAndThat()",
        "python method do_this_and_that()",
        "this is an example text. Does it contain any stop words? CamelCaseWord and snake_case_word",
    ]

    for text in texts:
        res = preprocess(text)
        LOGGER.info("Preprocess\n  from: %s\n  to  : %s", text, res)


def test_calculate_similarity():
    query1 = "hello john doe"
    query2 = "a document"

    corpus = [
        "this is a document",
        "this is john doe",
        "this is a document with john doe",
        "",
        "john john john doe doe doe document",
    ]

    doc_ids = ["d0", "d1", "d2", "d3", "d4"]
    sim = calculate_similarity(query1, corpus)

    LOGGER.info("Query: %s", query1)
    for s, doc in zip(sim, corpus):
        LOGGER.info("  sim = %.2f: %s ", s, doc)

    LOGGER.info("Queries: %s", [query1, query2])
    cos_sim = CosineSimilarity(zip(doc_ids, corpus))
    sim = cos_sim.get_similarities([query1, query2])
    LOGGER.info("cos sim for q0, q1 and all documents:\n %s", sim)
    LOGGER.info("cos sim only for q0: %s", sim[:, 0].flatten())

    docs = ["d0", "d2", "d4"]
    sim = cos_sim.get_similarities_by_doc_id([query1, query2], docs)
    LOGGER.info("cos sim for q0, q1 and docs %s:\n %s", ", ".join(docs), sim)


def test_cosine():
    queries = ["hello john doe", "a document"]

    docs = [
        ("d0", "this is a document"),
        ("d1", "this is john doe"),
        ("d2", "this is a document with john doe"),
        ("d3", ""),
        ("d4", "john john john doe doe doe document"),
        ("d5", "irrelevant text without matches doe document"),
    ]

    cos_sim = CosineSimilarity(docs)
    sim = cos_sim.get_similarities(queries)
    LOGGER.info("cos sim for q0, q1 and all documents:\n %s", sim)

    cos_sim = CosineSimilarity(docs[:3])
    sim = cos_sim.get_similarities(queries)
    LOGGER.info("cos sim for q0, q1 on smaller corpus:\n %s", sim)


# *******************************************************************
# MAIN
# *******************************************************************


if __name__ == "__main__":
    logging.basicConfig(
        format="%(name)s %(levelname)s %(message)s", level=logging.DEBUG
    )
    print(f"Hello from {__file__}")

    # download_nltk_data()
    test_preprocess()
    # test_calculate_similarity()
    # test_cosine()
