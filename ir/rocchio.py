import numpy as np


def rocchio(query, doc_rel, doc_nonrel, alpha=1.0, beta=0.75):
    """
    Parameters:
        - query = Query tf-idf vector.
        - doc_rel = List of tf-idf vectors marked relevant.
        - doc_nonrel = List of tf-idf vectors marked not relevant.
        - alpha = How much you keep of the original query. Bigger alpha -> don't drift too far from what the user typed.
        - beta = How strongly you move the query towards relevant items. Bigger beta -> pull the query toward what the user liked.

    Returns:
        - New query vector after weights.
    """
    q = np.array(query, dtype=float)

    # rel_centroid: Typical relevant document direction
    # nonrel_centroid: Typical non-relevant document direction.

    if len(doc_rel) > 0:
        rel_centroid = np.mean(np.array(doc_rel, dtype=float), axis=0)
    else:
        rel_centroid = np.zeros_like(q)

    if len(doc_nonrel) > 0:
        nonrel_centroid = np.mean(np.array(doc_nonrel, dtype=float), axis=0)
    else:
        nonrel_centroid = np.zeros_like(q)

    q_new = q + (alpha * rel_centroid) - (beta * nonrel_centroid)

    # Set negative weights to 0
    q_new = np.maximum(q_new, 0)

    return q_new