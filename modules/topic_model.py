import numpy as np

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import (
    CountVectorizer,
    ENGLISH_STOP_WORDS,
)
from umap import UMAP
from hdbscan import HDBSCAN


# ============================================================
# CUSTOM STOPWORDS
# ============================================================

CUSTOM_STOPWORDS = [

    # Reddit / web
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

    # Conversational
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

    # Contraction fragments
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
    Load a SentenceTransformer embedding model.
    """

    return SentenceTransformer(
        embedding_model_name
    )


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    documents,
    embedding_model,
    batch_size=64,
):
    """
    Generate embeddings for documents.
    """

    if documents is None or len(documents) == 0:

        raise ValueError(
            "No documents were provided for embedding."
        )


    embeddings = embedding_model.encode(
        list(documents),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


# ============================================================
# CREATE BERTopic MODEL
# ============================================================

def create_topic_model(
    min_topic_size=15,
    nr_topics="auto",
    ngram_min=1,
    ngram_max=2,
    top_n_words=20,
):
    """
    Create BERTopic model.

    Embeddings are supplied separately to fit_transform().
    """

    # --------------------------------------------------------
    # Stopwords
    # --------------------------------------------------------

    stop_words = set(
        ENGLISH_STOP_WORDS
    )

    stop_words.update(
        CUSTOM_STOPWORDS
    )

    stop_words = sorted(
        stop_words
    )


    # --------------------------------------------------------
    # Vectorizer
    # --------------------------------------------------------

    vectorizer_model = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(
            ngram_min,
            ngram_max,
        ),
        min_df=2,
        lowercase=True,
    )


    # --------------------------------------------------------
    # UMAP
    # --------------------------------------------------------

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )


    # --------------------------------------------------------
    # HDBSCAN
    # --------------------------------------------------------

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )


    # --------------------------------------------------------
    # BERTopic
    # --------------------------------------------------------

    topic_model = BERTopic(
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
    Run BERTopic and attach topic information to every document.

    Output document columns include:

        topic_id
        topic_auto_label
        topic_keywords
    """

    # --------------------------------------------------------
    # Copy dataframe
    # --------------------------------------------------------

    working_df = (
        df.copy()
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Documents
    # --------------------------------------------------------

    documents = (
        working_df[
            text_column
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(documents) == 0:

        raise ValueError(
            "No documents are available for topic modelling."
        )


    empty_positions = [
        i
        for i, document
        in enumerate(documents)
        if not document
    ]


    if empty_positions:

        raise ValueError(
            f"{len(empty_positions)} empty documents were found. "
            "Remove empty documents before running BERTopic."
        )


    if len(documents) < min_topic_size:

        raise ValueError(
            f"Only {len(documents):,} documents are available, "
            f"but min_topic_size={min_topic_size}."
        )


    # --------------------------------------------------------
    # Generate embeddings only when not supplied
    # --------------------------------------------------------

    if embeddings is None:

        if embedding_model is None:

            embedding_model = (
                load_embedding_model(
                    embedding_model_name
                )
            )


        embeddings = create_embeddings(
            documents=documents,
            embedding_model=embedding_model,
            batch_size=batch_size,
        )


    # --------------------------------------------------------
    # Validate embeddings
    # --------------------------------------------------------

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )


    if embeddings.ndim != 2:

        raise ValueError(
            "Embeddings must be a two-dimensional array."
        )


    if len(embeddings) != len(documents):

        raise ValueError(
            "Document and embedding counts do not match. "
            f"Documents={len(documents):,}; "
            f"Embeddings={len(embeddings):,}."
        )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    topic_model = create_topic_model(
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        top_n_words=top_n_words,
    )


    # --------------------------------------------------------
    # Fit model
    # --------------------------------------------------------

    topics, probabilities = (
        topic_model.fit_transform(
            documents,
            embeddings=embeddings,
        )
    )


    # --------------------------------------------------------
    # Topic ID for every document
    # --------------------------------------------------------

    working_df[
        "topic_id"
    ] = topics


    # --------------------------------------------------------
    # Topic information
    # --------------------------------------------------------

    topic_info = (
        topic_model.get_topic_info()
        .copy()
    )


    # ========================================================
    # BUILD TOPIC KEYWORD + AUTO LABEL MAPS
    # ========================================================

    topic_keyword_map = {}

    topic_auto_label_map = {}


    for topic_id in topic_info["Topic"]:

        topic_id = int(topic_id)


        # ----------------------------------------------------
        # Outliers
        # ----------------------------------------------------

        if topic_id == -1:

            topic_keyword_map[
                topic_id
            ] = "OUTLIER / UNASSIGNED"

            topic_auto_label_map[
                topic_id
            ] = "OUTLIER / UNASSIGNED"

            continue


        # ----------------------------------------------------
        # Topic keywords
        # ----------------------------------------------------

        topic_words = (
            topic_model.get_topic(
                topic_id
            )
        )


        if topic_words:

            keywords = [
                word
                for word, score
                in topic_words[
                    :top_n_words
                ]
            ]


            topic_keyword_map[
                topic_id
            ] = ", ".join(
                keywords
            )


            # Short automatic topic label
            topic_auto_label_map[
                topic_id
            ] = " | ".join(
                keywords[:5]
            )


        else:

            topic_keyword_map[
                topic_id
            ] = ""

            topic_auto_label_map[
                topic_id
            ] = (
                f"Topic {topic_id}"
            )


    # ========================================================
    # ADD LABEL TO EACH DOCUMENT
    # ========================================================

    working_df[
        "topic_auto_label"
    ] = (
        working_df[
            "topic_id"
        ]
        .map(
            topic_auto_label_map
        )
    )


    working_df[
        "topic_keywords"
    ] = (
        working_df[
            "topic_id"
        ]
        .map(
            topic_keyword_map
        )
    )


    # ========================================================
    # ADD TO TOPIC SUMMARY
    # ========================================================

    topic_info[
        "Auto_Label"
    ] = (
        topic_info[
            "Topic"
        ]
        .map(
            topic_auto_label_map
        )
    )


    topic_info[
        "Topic_Keywords"
    ] = (
        topic_info[
            "Topic"
        ]
        .map(
            topic_keyword_map
        )
    )


    topic_info[
        "Topic_Status"
    ] = (
        topic_info[
            "Topic"
        ]
        .apply(
            lambda topic:
            "OUTLIER / UNASSIGNED"
            if topic == -1
            else "TOPIC"
        )
    )


    # ========================================================
    # RETURN
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

        "topic_keyword_map":
            topic_keyword_map,

        "topic_auto_label_map":
            topic_auto_label_map,
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
    Return topic words and c-TF-IDF scores.
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
    Return representative documents for one topic.
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
