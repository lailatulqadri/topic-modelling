# ============================================================
# TOPIC MODEL MODULE
# Dynamic Topic Modelling System
# Optimised for Streamlit / BERTopic
# ============================================================

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import (
    CountVectorizer,
    ENGLISH_STOP_WORDS,
)

from umap import UMAP
from hdbscan import HDBSCAN

import numpy as np


# ============================================================
# CUSTOM STOPWORDS
# ============================================================

CUSTOM_STOPWORDS = [

    # --------------------------------------------------------
    # Reddit / Web
    # --------------------------------------------------------

    "reddit",
    "post",
    "posts",
    "comment",
    "comments",
    "thread",
    "subreddit",

    "http",
    "https",
    "www",
    "com",
    "amp",

    "deleted",
    "removed",

    # --------------------------------------------------------
    # General conversational words
    # --------------------------------------------------------

    "like",
    "just",
    "really",
    "know",
    "think",

    "said",
    "say",

    "people",
    "person",

    "thing",
    "things",

    "got",
    "get",
    "getting",

    "going",
    "went",

    "want",
    "wanted",

    "feel",
    "felt",

    "actually",
    "maybe",
    "probably",

    "yeah",
    "yes",
    "no",

    "hi",
    "hello",

    "thanks",
    "thank",

    # --------------------------------------------------------
    # Contraction fragments
    # --------------------------------------------------------

    "ve",
    "ll",
    "don",
    "didn",
    "doesn",
    "isn",
    "wasn",
    "weren",
    "wouldn",
    "couldn",
    "shouldn",
]


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model(
    embedding_model_name="all-MiniLM-L6-v2",
):

    """
    Load SentenceTransformer model.

    Streamlit should cache this function using st.cache_resource
    from app.py.
    """

    model = SentenceTransformer(
        embedding_model_name
    )

    return model


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    documents,
    embedding_model,
    batch_size=64,
):

    """
    Generate document embeddings separately from BERTopic.

    This prevents BERTopic from automatically generating
    embeddings every time fit_transform() is called.

    Parameters
    ----------
    documents : list
        List of text documents.

    embedding_model : SentenceTransformer
        Loaded SentenceTransformer model.

    batch_size : int
        Number of documents processed per batch.

    Returns
    -------
    numpy.ndarray
        Document embeddings.
    """

    embeddings = embedding_model.encode(

        documents,

        batch_size=batch_size,

        show_progress_bar=True,

        normalize_embeddings=True,

        convert_to_numpy=True,
    )

    return embeddings


# ============================================================
# CREATE TOPIC MODEL
# ============================================================

def create_topic_model(
    min_topic_size=15,
    nr_topics="auto",
    ngram_min=1,
    ngram_max=2,
    top_n_words=20,
):

    """
    Create and configure BERTopic.

    IMPORTANT:
    Embeddings are generated separately.
    BERTopic therefore does NOT need to run
    SentenceTransformer internally.
    """

    # ========================================================
    # STOPWORDS
    # ========================================================

    stop_words = set(
        ENGLISH_STOP_WORDS
    )

    stop_words.update(
        CUSTOM_STOPWORDS
    )

    stop_words = sorted(
        stop_words
    )


    # ========================================================
    # VECTORIZER
    # ========================================================

    vectorizer_model = CountVectorizer(

        stop_words=stop_words,

        ngram_range=(
            ngram_min,
            ngram_max
        ),

        min_df=2,

        lowercase=True,
    )


    # ========================================================
    # UMAP
    # ========================================================

    umap_model = UMAP(

        n_neighbors=15,

        n_components=5,

        min_dist=0.0,

        metric="cosine",

        random_state=42,

        low_memory=True,
    )


    # ========================================================
    # HDBSCAN
    # ========================================================

    hdbscan_model = HDBSCAN(

        min_cluster_size=min_topic_size,

        metric="euclidean",

        cluster_selection_method="eom",

        prediction_data=True,
    )


    # ========================================================
    # BERTopic
    # ========================================================

    topic_model = BERTopic(

        # IMPORTANT:
        # No SentenceTransformer here.
        # Embeddings are passed directly to fit_transform().

        embedding_model=None,

        vectorizer_model=vectorizer_model,

        umap_model=umap_model,

        hdbscan_model=hdbscan_model,

        top_n_words=top_n_words,

        nr_topics=nr_topics,

        calculate_probabilities=False,

        verbose=True,
    )


    return topic_model


# ============================================================
# RUN TOPIC MODEL
# ============================================================

def run_topic_model(
    df,
    text_column="_topic_text",
    embedding_model_name="all-MiniLM-L6-v2",
    min_topic_size=15,
    nr_topics="auto",
    ngram_min=1,
    ngram_max=2,
    top_n_words=20,
    embedding_model=None,
    embeddings=None,
    batch_size=64,
):

    """
    Run BERTopic on a dataframe.

    Embeddings can optionally be supplied.

    This allows Streamlit to cache embeddings so that
    BERTopic does not regenerate them unnecessarily.

    Returns
    -------
    dict containing:

        model
        data
        topic_info
        documents
        topics
        probabilities
        embeddings
    """


    # ========================================================
    # COPY DATA
    # ========================================================

    working_df = df.copy()


    # ========================================================
    # PREPARE DOCUMENTS
    # ========================================================

    documents = (

        working_df[
            text_column
        ]

        .fillna("")

        .astype(str)

        .str.strip()

        .tolist()
    )


    # ========================================================
    # REMOVE EMPTY DOCUMENTS
    # ========================================================

    valid_mask = [
        bool(doc)
        for doc in documents
    ]

    if not all(valid_mask):

        working_df = (
            working_df
            .loc[valid_mask]
            .copy()
            .reset_index(drop=True)
        )

        documents = [
            doc
            for doc in documents
            if doc
        ]


    # ========================================================
    # CHECK DATA
    # ========================================================

    if len(documents) < min_topic_size:

        raise ValueError(
            f"Not enough documents for topic modelling. "
            f"Found {len(documents)} documents but "
            f"min_topic_size={min_topic_size}."
        )


    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================

    if embeddings is None:

        if embedding_model is None:

            embedding_model = (
                load_embedding_model(
                    embedding_model_name
                )
            )


        # ====================================================
        # CREATE EMBEDDINGS
        # ====================================================

        embeddings = (
            create_embeddings(

                documents=documents,

                embedding_model=embedding_model,

                batch_size=batch_size,
            )
        )


    # ========================================================
    # VALIDATE EMBEDDINGS
    # ========================================================

    embeddings = np.asarray(
        embeddings
    )


    if len(embeddings) != len(documents):

        raise ValueError(

            "Number of embeddings does not match "
            "number of documents. "

            f"Documents: {len(documents)}, "
            f"Embeddings: {len(embeddings)}"
        )


    # ========================================================
    # CREATE BERTopic MODEL
    # ========================================================

    topic_model = create_topic_model(

        min_topic_size=min_topic_size,

        nr_topics=nr_topics,

        ngram_min=ngram_min,

        ngram_max=ngram_max,

        top_n_words=top_n_words,
    )


    # ========================================================
    # FIT BERTopic
    # ========================================================

    topics, probabilities = (
        topic_model.fit_transform(

            documents,

            embeddings=embeddings,
        )
    )


    # ========================================================
    # ADD TOPIC ID
    # ========================================================

    working_df[
        "topic_id"
    ] = topics


    # ========================================================
    # GET TOPIC INFORMATION
    # ========================================================

    topic_info = (
        topic_model.get_topic_info()
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "model":
            topic_model,

        "data":
            working_df,

        "topic_info":
            topic_info,

        "documents":
            documents,

        "topics":
            topics,

        "probabilities":
            probabilities,

        "embeddings":
            embeddings,
    }


# ============================================================
# GET TOPIC KEYWORDS
# ============================================================

def get_topic_keywords(
    topic_model,
    topic_id,
    top_n=20,
):

    """
    Return keywords and c-TF-IDF scores
    for a selected topic.
    """

    try:

        topic = (
            topic_model.get_topic(
                topic_id
            )
        )

        if not topic:

            return []

        return topic[:top_n]

    except Exception:

        return []


# ============================================================
# GET REPRESENTATIVE DOCUMENTS
# ============================================================

def get_representative_documents(
    topic_model,
    topic_id,
):

    """
    Return representative documents
    for a selected BERTopic topic.
    """

    try:

        documents = (
            topic_model
            .get_representative_docs(
                topic_id
            )
        )

        if documents is None:

            return []

        return documents

    except Exception:

        return []
