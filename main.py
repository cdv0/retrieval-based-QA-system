from pathlib import Path
import pandas as pd

from ir.preprocess import preprocess
from ir.vectorize import vectorize

if __name__ == "__main__":
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
    vocab, matrix = vectorize(docs_id, tokens)

    # Label matrix
    df = pd.DataFrame(matrix, index=docs_id, columns=vocab)

    # Round values to 3 decimal points and export as csv
    df.round(3).to_csv("tfidf_matrix_rounded.csv", index=True)
    print(df.round(3))
