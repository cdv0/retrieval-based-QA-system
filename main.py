from pathlib import Path
import pandas as pd
import numpy as np

from ir.preprocess import preprocess
from ir.vectorize import vectorize
from ir.cosine_similarity import cosine_similarity

if __name__ == "__main__":
    # === Cosine similarity on corpus ===
    corpus_dir = Path("corpus")
    txt_files = sorted(corpus_dir.glob("*.txt"))

    docs_id = []
    tokens = []

    # Build doc ids using filenames and tokens (2d list)
    for fp in txt_files:
        text = fp.read_text(encoding="utf-8")
        docs_id.append(fp.name)
        tokens.append(preprocess(text))

    # Vectorize and create the tf-idf matrix
    vocab, doc_matrix, idf_map = vectorize(docs_id, tokens)

    # Label matrix
    df = pd.DataFrame(doc_matrix, index=docs_id, columns=vocab)

    # Round values to 3 decimal points and export as csv
    df.round(3).to_csv("doc_tfidf_matrix_rounded.csv", index=True)
    print(df.round(3))

    # Compute cosine similarity against the query and corpus
    query = "What does the immune system do?"
    tokens_query = preprocess(query)
    query_matrix = np.zeros(len(vocab))  # Needs to be the same legnth to compute the dot product in cosine similarity
    for j, term in enumerate(vocab):
        query_matrix[j] = tokens_query.count(term) * idf_map[term]  # tf-idf for query

    scores = []
    for i, docname in enumerate(docs_id):
        score = cosine_similarity(query_matrix, doc_matrix[i])
        scores.append((docname, score))

    scores.sort(key=lambda x: x[1], reverse=True)  # Sort the score part by highest score to lowest score
    print(scores)


    # === Cosine similarity on the document with the highest score to get best sentence answer to query ===
