import numpy as np

def cosine_similarity(d1_or_query, d2):
    """
    Parameters:
        - d1_or_query (list): The first document/query to compare. The list of tf-idf values in this document.
        - d2 (list): The second document to compare. A list of tf-idf values in this document.

    Returns:
        - Returns a similarity score based on how similar the two documents are.
        Value closer to 1 means the documents are more similar.
    """
    euclidean_length_d1 = np.sqrt(sum(x**2 for x in d1_or_query))
    euclidean_length_d2 = np.sqrt(sum(x**2 for x in d2))

    if euclidean_length_d1 == 0 or euclidean_length_d2 == 0:
        return 0.0

    euclidean_normalize_d1 = np.divide(d1_or_query, euclidean_length_d1)
    euclidean_normalize_d2 = np.divide(d2, euclidean_length_d2)

    return float(np.dot(euclidean_normalize_d1, euclidean_normalize_d2))