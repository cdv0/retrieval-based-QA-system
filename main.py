from pathlib import Path
import pandas as pd
import numpy as np

from ir.preprocess import preprocess
from ir.vectorize import vectorize
from ir.cosine_similarity import cosine_similarity

if __name__ == "__main__":
    # 1. === Cosine similarity on corpus ===
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
    query = "What is immunology?"
    tokens_query = preprocess(query)
    query_matrix = np.zeros(len(vocab))  # Needs to be the same length to compute the dot product in cosine similarity
    for j, term in enumerate(vocab):
        query_matrix[j] = tokens_query.count(term) * idf_map[term]  # tf-idf for query

    scores = []
    for i, doc_name in enumerate(docs_id):
        score = cosine_similarity(query_matrix, doc_matrix[i])
        scores.append((doc_name, score))

    scores.sort(key=lambda x: x[1], reverse=True)  # Sort the score part by highest score to lowest score
    print(scores)



    # 2. === Cosine similarity on the document with the highest score to get best sentence answer to query ===

    # Get the top document's sentences (each one is its own doc)
    top_doc_name = scores[0][0]
    top_doc_path = corpus_dir / top_doc_name
    top_doc_text = top_doc_path.read_text(encoding="utf-8")
    top_doc_sentences = top_doc_text.split(". ")
    print(top_doc_sentences)

    sentence_ids = [i for i in range(len(top_doc_sentences))]
    top_tokens = []

    # Create 2d matrix with processed sentences
    for i in top_doc_sentences:
        top_tokens.append(preprocess(i))

    # Vectorize and create the tf-idf matrix
    top_doc_vocab, top_doc_matrix, top_doc_idf_map = vectorize(sentence_ids, top_tokens)

    # Label matrix
    top_doc_df = pd.DataFrame(top_doc_matrix, index=sentence_ids, columns=top_doc_vocab)

    # Round values to 3 decimal points and export as csv
    top_doc_df.round(3).to_csv("top_doc_tfidf_matrix_rounded.csv", index=True)
    print(top_doc_df.round(3))

    # Compute cosine similarity against the query and corpus
    # query = "What does the immune system do?"
    # tokens_query = preprocess(query)
    top_doc_query_matrix = np.zeros(len(top_doc_vocab))  # Needs to be the same length to compute the dot product in cosine similarity
    for j, term in enumerate(top_doc_vocab):
        top_doc_query_matrix[j] = tokens_query.count(term) * top_doc_idf_map.get(term, 0.0)  # tf-idf for query

    top_sentence_scores = []
    for i, sentence_index in enumerate(sentence_ids):
        top_sentence_score = cosine_similarity(top_doc_query_matrix, top_doc_matrix[i])
        top_sentence_scores.append((sentence_index, top_sentence_score))

    top_sentence_scores.sort(key=lambda x: x[1], reverse=True)  # Sort the score part by highest score to lowest score
    print(top_sentence_scores)
    print(top_doc_sentences[top_sentence_scores[0][0]])