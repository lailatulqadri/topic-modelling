# ============================================================
# TOPIC MODEL MODULE
# Dynamic Topic Modelling System
# ============================================================

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
#
# Add/remove your own stopwords here.
# These words will NOT appear as BERTopic keywords.
#
# IMPORTANT:
# These stopwords affect topic representation.
# They do not delete words from your original dataset.
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
# CREATE TOPIC MODEL
# ============================================================

def create_topic_model(
    embedding_model_name="all-MiniLM-L6-v2",
    min_topic_size=15,
    nr_topics="auto",
    ngram_min=1,
    ngram_max=2,
    top_n_words=20,
):

    """
    Create and configure BERTopic.

    Parameters
    ----------
    embedding_model_name : str
        SentenceTransformer model.

    min_topic_size : int
        Minimum number of documents required to form a topic.

    nr_topics : str/int
        Number of topics.
        "auto" lets BERTopic reduce similar topics automatically.

    ngram_min : int
        Minimum n-gram size.

    ngram_max : int
        Maximum n-gram size.

    top_n_words : int
        Number of keywords returned for each topic.
    """


    # ========================================================
    # EMBEDDING MODEL
    # ========================================================

    embedding_model = SentenceTransformer(
        embedding_model_name
    )


    # ========================================================
    # STOPWORDS
    # ========================================================

    # Start with sklearn English stopwords
    stop_words = set(
        ENGLISH_STOP_WORDS
    )


    # Add our manually defined stopwords
    stop_words.update(
        CUSTOM_STOPWORDS
    )


    # CountVectorizer expects a list
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

        embedding_model=embedding_model,

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
):

    """
    Run BERTopic on a dataframe.

    Returns:
        model
        dataframe with topic_id
        topic information
        documents
        topic assignments
        probabilities
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
    # CREATE MODEL
    # ========================================================

    topic_model = create_topic_model(

        embedding_model_name=(
            embedding_model_name
        ),

        min_topic_size=(
            min_topic_size
        ),

        nr_topics=(
            nr_topics
        ),

        ngram_min=(
            ngram_min
        ),

        ngram_max=(
            ngram_max
        ),

        top_n_words=(
            top_n_words
        ),
    )


    # ========================================================
    # FIT MODEL
    # ========================================================

    topics, probabilities = (
        topic_model.fit_transform(
            documents
        )
    )


    # ========================================================
    # ADD TOPIC ID TO DATAFRAME
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


        return topic[
            :top_n
        ]


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