import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

from ir.preprocess import preprocess
from ir.vectorize import vectorize
from ir.cosine_similarity import cosine_similarity


# Override grey text when text area is disabled
st.markdown(
    """
    <style>
    textarea[disabled] {
        color: black !important;
        -webkit-text-fill-color: black !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def run_query(query, corpus_folder="corpus"):
    # 1. === Cosine similarity on corpus ===
    corpus_dir = Path(corpus_folder)
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

    # Compute cosine similarity against the query and corpus
    tokens_query = preprocess(query)
    query_matrix = np.zeros(len(vocab))  # Needs to be the same length to compute the dot product in cosine similarity
    for j, term in enumerate(vocab):
        query_matrix[j] = tokens_query.count(term) * idf_map[term]  # tf-idf for query

    scores = []
    for i, doc_name in enumerate(docs_id):
        score = cosine_similarity(query_matrix, doc_matrix[i])
        scores.append((doc_name, score))

    scores.sort(key=lambda x: x[1], reverse=True)  # Sort the score part by highest score to lowest score

    # 2. === Cosine similarity on the document with the highest score to get best sentence answer to query ===

    # Get the top document's sentences (each one is its own doc)
    top_doc_name = scores[0][0]
    top_doc_path = corpus_dir / top_doc_name
    top_doc_text = top_doc_path.read_text(encoding="utf-8")
    top_doc_sentences = top_doc_text.split(". ")

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

    # Compute cosine similarity against the query and corpus
    top_doc_query_matrix = np.zeros(len(top_doc_vocab))  # Needs to be the same length to compute the dot product in cosine similarity
    for j, term in enumerate(top_doc_vocab):
        top_doc_query_matrix[j] = tokens_query.count(term) * top_doc_idf_map.get(term, 0.0)  # tf-idf for query

    top_sentence_scores = []
    for i, sentence_index in enumerate(sentence_ids):
        top_sentence_score = cosine_similarity(top_doc_query_matrix, top_doc_matrix[i])
        top_sentence_scores.append((sentence_index, top_sentence_score))

    top_sentence_scores.sort(key=lambda x: x[1], reverse=True)  # Sort the score part by highest score to lowest score

    answer = top_doc_sentences[top_sentence_scores[0][0]]

    # Return data for Streamlit
    return {
        "query": query,
        "scores": scores,
        "top_doc_name": top_doc_name,
        "top_doc_sentences": top_doc_sentences,
        "top_sentence_scores": top_sentence_scores,
        "answer": answer,
    }


# STREAMLIT CONFIG
st.set_page_config(initial_sidebar_state="expanded")


# SESSION/CHAT STATE
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! 👋🏻"}]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""


# SIDEBAR
with st.sidebar:
    st.title("Controls")

    # MODE
    mode = st.selectbox("Select a mode:", ["Cosine", "Rocchio", "Both"])

    st.divider()

    # PRESET QUERIES
    st.subheader("Recommended queries")
    presets = [
        "What is immunology?",
        "What is the innate immune system?",
        "What barriers are used for protection in innate immunity?",
        "What is genetics?",
        "How many genes do humans have?",
        "What is the digestive system made up of?",
    ]

    for i, preset in enumerate(presets, start=1):
        if st.button(f"{i}. {preset}", key=f"preset_{i}", use_container_width=True):
            st.session_state.pending_query = preset

    st.divider()

    # CORPUS BROWSER
    st.subheader("Corpus browser")

    corpus_dir = Path("corpus")
    txt_files = sorted(corpus_dir.glob("*.txt"))
    file_names = [fp.name for fp in txt_files]

    selected = st.selectbox("Select a document", file_names)

    if selected:
        doc_text = (corpus_dir / selected).read_text(encoding="utf-8")
        st.text_area("Document text", doc_text, height=350, disabled=True)

    st.divider()

    # CLEAR CHAT
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Hi there! 👋🏻"}]
        st.session_state.pending_query = ""
        st.toast("Chat has been cleared", icon="✅", duration=3)
        st.rerun()

# TITLE
st.title("Retrieval-based QA System")


# DISPLAY CHAT
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if "top_sentence_scores" in message:
                with st.expander("Top sentence matches"):
                    st.markdown("**Top sentence matches in the top document**")
                    for sent_idx, score in message["top_sentence_scores"]:
                        st.markdown(f"- `{score:.4f}` - {message['top_doc_sentences'][sent_idx]}")

            elif "cosine_top_sentence_scores" in message:
                with st.expander("Top sentence matches (Cosine)"):
                    st.markdown("**Top sentence matches in the top document (Cosine)**")
                    for sent_idx, score in message["cosine_top_sentence_scores"]:
                        st.markdown(f"- `{score:.4f}` - {message['cosine_top_doc_sentences'][sent_idx]}")


# TEXT INPUT
prompt = st.chat_input("Ask about the immune system, genetics, or the digestive system!")
pending = st.session_state.get("pending_query", "")
new_query = ""

if prompt:
    new_query = prompt
elif pending:
    new_query = pending
    st.session_state.pending_query = ""

if new_query:
    st.session_state.messages.append({"role": "user", "content": new_query})

    try:
        if mode == "Cosine":
            out = run_query(new_query, corpus_folder="corpus")

            assistant_content = f"### {mode}\n\n"
            assistant_content += f"**Answer:** {out['answer'] if out['answer'] else '(no answer found)'}\n\n"
            assistant_content += f"Top document: `{out['top_doc_name']}`\n\n"
            assistant_content += "**Top documents:**\n"
            for i, (doc_name, score) in enumerate(out["scores"][:5], start=1):
                assistant_content += f"- **#{i}. {doc_name}**  -  (score: `{score:.4f}`)\n"

            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_content,
                "top_sentence_scores": out["top_sentence_scores"][:5],
                "top_doc_sentences": out["top_doc_sentences"],
            })

        elif mode == "Rocchio":
            assistant_content = f"### {mode}\n\n"
            assistant_content += "Rocchio not hooked up yet."
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_content,
            })


        else:  # Both
            out = run_query(new_query, corpus_folder="corpus")

            # Cosine message
            cosine_block = f"### Cosine\n**Answer:** {out['answer']}\n\nTop doc: {out['top_doc_name']}\n"
            cosine_block += "\n**Top documents:**\n"
            for i, (doc_name, score) in enumerate(out["scores"][:5], start=1):
                cosine_block += f"- **#{i}. {doc_name}** - (score: {score:.4f})\n"
            st.session_state.messages.append({
                "role": "assistant",
                "content": cosine_block,
                "cosine_top_sentence_scores": out["top_sentence_scores"][:5],
                "cosine_top_doc_sentences": out["top_doc_sentences"],
            })

            # Rocchio message
            st.session_state.messages.append({
                "role": "assistant",
                "content": "### Rocchio\n(coming soon)\n"
            })

    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

    st.rerun()