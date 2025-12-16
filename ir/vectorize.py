import math
import numpy as np


def tf(t, d):
    """
    tf: Term Frequency

    Parameters:
        - t: Term
        - d: Document, represented as a list of tokens.
    Returns:
        - The number of times term t appears in document d.
    """
    return d.count(t)


def idf(t, docs):
    """
    idf: Inverse Document Frequency. Gives common terms less weight and rare terms more weight.

    Parameters:
        - t: Term
        - N: Total number of documents
        - df_t: Number of documents that contain term t
        - docs: Set of documents. 2D list. Ex. [set(doc1), set(doc2), ...]

    Returns:
        A weight for term t based on term relevance.
    """
    N = len(docs)
    df_t = sum(1 for d in docs if t in d)
    return math.log(N / df_t) if df_t else 0.0  # Default to 0 if term isn't found in any doc


def tf_idf(tf, idf):
    """
    Parameters:
        - tf: Term frequency.
        - idf: Inverse document frequency. Gives common terms less weight and rare terms more weight.

    Returns:
        - Measures how important a term is in a document.
    """
    return tf * idf


def vectorize(docs_id, tokens):
    """
    Parameters:
        - docs_id: List of document ID.
        - tokens: List of all tokens. 2D list.

    Returns:
        - A matrix of doc x unique terms and store each term's tf-idf value.
    """
    concat_tokens = []
    for token in tokens:
        concat_tokens.extend(token)
    unique_sorted_tokens = sorted(set(concat_tokens))  # use for matrix x value

    rows = len(docs_id)
    cols = len(unique_sorted_tokens)

    matrix = np.zeros((rows, cols))

    # Compute tf-idf and store value for each cell
    doc_sets = [set(token) for token in tokens]
    idf_map = {term: idf(term, doc_sets) for term in unique_sorted_tokens}

    for i, doc in enumerate(tokens):
        for j, term in enumerate(unique_sorted_tokens):
            matrix[i, j] = tf_idf(doc.count(term), idf_map[term])

    return unique_sorted_tokens, matrix, idf_map