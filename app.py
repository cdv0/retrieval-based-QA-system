import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

from ir.preprocess import preprocess
from ir.vectorize import vectorize
from ir.cosine_similarity import cosine_similarity
from ir.rocchio import rocchio


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


def build_sentence_data(query, corpus_dir, doc_name):
    top_doc_path = corpus_dir / doc_name
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
    tokens_query = preprocess(query)
    top_doc_query_matrix = np.zeros(len(top_doc_vocab))  # Needs to be the same length to compute the dot product in cosine similarity
    for j, term in enumerate(top_doc_vocab):
        top_doc_query_matrix[j] = tokens_query.count(term) * top_doc_idf_map.get(term, 0.0)  # tf-idf for query

    top_sentence_scores = []
    for i, sentence_index in enumerate(sentence_ids):
        top_sentence_score = cosine_similarity(top_doc_query_matrix, top_doc_matrix[i])
        top_sentence_scores.append((sentence_index, top_sentence_score))

    top_sentence_scores.sort(key=lambda x: x[1], reverse=True)  # Sort the score part by highest score to lowest score

    answer = top_doc_sentences[top_sentence_scores[0][0]] if len(top_doc_sentences) > 0 else ""

    return {
        "top_doc_name": doc_name,
        "top_doc_sentences": top_doc_sentences,
        "top_sentence_scores": top_sentence_scores,
        "answer": answer,
        "top_doc_vocab": top_doc_vocab,
        "top_doc_matrix": top_doc_matrix,
        "top_doc_query_matrix": top_doc_query_matrix,
    }


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
    sent = build_sentence_data(query, corpus_dir, top_doc_name)

    # Return data for Streamlit
    return {
        "query": query,
        "scores": scores,
        "top_doc_name": sent["top_doc_name"],
        "top_doc_sentences": sent["top_doc_sentences"],
        "top_sentence_scores": sent["top_sentence_scores"],
        "answer": sent["answer"],
        "corpus_dir": corpus_dir,
        "top_doc_vocab": sent["top_doc_vocab"],
        "top_doc_matrix": sent["top_doc_matrix"],
        "top_doc_query_matrix": sent["top_doc_query_matrix"],
    }


# STREAMLIT CONFIG
st.set_page_config(initial_sidebar_state="expanded")


# SESSION/CHAT STATE
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! 👋🏻"}]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

if "last_out" not in st.session_state:  # Stores the most recent output from run_query() (Cosine baseline) like sentence tf-idf matrix, original query tf-idf, basline cosine sentence scores, and sentences themselves.
    st.session_state.last_out = None

if "sent_rel" not in st.session_state:  # Stores the current list of relevant sentences.
    st.session_state.sent_rel = []

if "rocchio_sent_out" not in st.session_state:  # Stores the Rocchio re-ranked sentence results
    st.session_state.rocchio_sent_out = None


# SIDEBAR
with st.sidebar:
    st.title("Controls")

    # MODE
    mode = st.selectbox("Select a mode:", ["Cosine", "Rocchio"])

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
        st.session_state.last_out = None
        st.session_state.sent_rel = []
        st.session_state.rocchio_sent_out = None
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
                with st.expander("Top sentence matches (Cosine)"):
                    st.markdown("**Top sentence matches in the top document (Cosine)**")
                    for sent_idx, score in message["top_sentence_scores"]:
                        st.markdown(f"- `{score:.4f}` - {message['top_doc_sentences'][sent_idx]}")

            elif "rocchio_top_sentence_scores" in message:
                with st.expander("Top sentence matches (Rocchio)"):
                    st.markdown("**Top sentence matches in the top document (Rocchio)**")
                    for sent_idx, score in message["rocchio_top_sentence_scores"]:
                        st.markdown(f"- `{score:.4f}` - {message['rocchio_top_doc_sentences'][sent_idx]}")


if st.session_state.last_out is not None and mode == "Rocchio":
    out_base = st.session_state.last_out
    sent_scores = out_base["top_sentence_scores"][:5]

    st.subheader("Select relevant sentence documents")

    sent_options = []
    for idx, sc in sent_scores:
        txt = out_base["top_doc_sentences"][idx] if idx < len(out_base["top_doc_sentences"]) else ""
        short = (txt[:90] + "...") if len(txt) > 90 else txt  # Truncate the text if longer than 90 characters
        sent_options.append(f"{idx}. {short}")

    st.session_state.sent_rel = st.multiselect(
        "Relevant sentence docs",
        sent_options,
        default=[s for s in st.session_state.sent_rel if s in sent_options],
        key="sent_rel_picker",
    )

    if st.button("Apply Rocchio (Sentences)", width="stretch"):
        top_doc_matrix = out_base["top_doc_matrix"]
        top_doc_query_matrix = out_base["top_doc_query_matrix"]

        def parse_idx(x):
            try:
                return int(x.split(". ")[0].strip())
            except:
                return None

        rel_idx = [parse_idx(x) for x in st.session_state.sent_rel]
        rel_idx = [i for i in rel_idx if i is not None and i >= 0 and i < len(top_doc_matrix)]  # Validate sentence indexes

        rel_vecs = [top_doc_matrix[i] for i in rel_idx]  # List of vectors for those sentences. Used to plug into Rocchio algorithm

        q2_new = rocchio(top_doc_query_matrix, rel_vecs, [], alpha=1.0, beta=0.75)

        new_sent_scores = []
        for i in range(len(top_doc_matrix)):
            sc = cosine_similarity(q2_new, top_doc_matrix[i])
            new_sent_scores.append((i, sc))

        new_sent_scores.sort(key=lambda x: x[1], reverse=True)

        answer2 = out_base["top_doc_sentences"][new_sent_scores[0][0]] if len(new_sent_scores) > 0 else ""

        st.session_state.rocchio_sent_out = {
            "top_doc_name": out_base["top_doc_name"],
            "top_doc_sentences": out_base["top_doc_sentences"],
            "top_sentence_scores": new_sent_scores,
            "answer": answer2,
        }

        assistant_content = "### Rocchio (Sentences)\n\n"
        assistant_content += f"**Answer:** {st.session_state.rocchio_sent_out['answer'] if st.session_state.rocchio_sent_out['answer'] else '(no answer found)'}\n\n"
        assistant_content += f"Top document: `{st.session_state.rocchio_sent_out['top_doc_name']}`\n\n"

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_content,
            "rocchio_top_sentence_scores": st.session_state.rocchio_sent_out["top_sentence_scores"][:5],
            "rocchio_top_doc_sentences": st.session_state.rocchio_sent_out["top_doc_sentences"],
        })

        st.rerun()


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
            st.session_state.last_out = out

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
            out = run_query(new_query, corpus_folder="corpus")
            st.session_state.last_out = out
            st.session_state.sent_rel = []
            st.session_state.rocchio_sent_out = None

            assistant_content = "### Cosine (Baseline)\n\n"
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

            st.session_state.messages.append({
                "role": "assistant",
                "content": "### Rocchio\nSelect relevant sentence docs below to refine sentence ranking.\n"
            })

    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

    st.rerun()