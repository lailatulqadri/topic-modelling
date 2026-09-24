# ============================================================
# DYNAMIC TOPIC MODELLING SYSTEM
# Optimised Streamlit + BERTopic Version
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ============================================================
# MODULE IMPORTS
# ============================================================

from modules.data_loader import (
    load_csv,
    get_column_summary,
    detect_text_columns,
    detect_categorical_columns,
)

from modules.filters import (
    prepare_text_data,
    get_filter_values,
)

from modules.topic_model import (
    run_topic_model,
    get_topic_keywords,
    get_representative_documents,
    load_embedding_model,
    create_embeddings,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dynamic Topic Modelling",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CACHED EMBEDDING FUNCTIONS
# ============================================================

@st.cache_resource
def get_cached_embedding_model(
    model_name,
):
    """
    Load SentenceTransformer once and keep it in memory.
    """

    return load_embedding_model(
        model_name
    )


@st.cache_data(
    show_spinner=False,
    max_entries=3,
)
def generate_cached_embeddings(
    documents,
    model_name,
):
    """
    Generate embeddings for the complete valid corpus.

    Streamlit caches the result using:
        documents
        model_name

    If both remain unchanged, embeddings are reused.
    """

    model = get_cached_embedding_model(
        model_name
    )

    embeddings = create_embeddings(
        documents=list(documents),
        embedding_model=model,
        batch_size=64,
    )

    return embeddings


# ============================================================
# HEADER
# ============================================================

st.title(
    "Dynamic Topic Modelling System"
)

st.caption(
    "Explore topics dynamically across relevance, sentiment, "
    "emotion, source, time, and other dataset categories."
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "data" not in st.session_state:
    st.session_state.data = None

if "filter_count" not in st.session_state:
    st.session_state.filter_count = 1


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Dataset"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
    )


# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a CSV file from the sidebar to begin."
    )

    st.stop()


try:

    df, encoding = load_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to load the dataset:\n\n{e}"
    )

    st.stop()


st.session_state.data = df


# ============================================================
# 1. DATASET INFORMATION
# ============================================================

st.subheader(
    "1. Dataset Overview"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Rows",
        f"{len(df):,}",
    )


with col2:

    st.metric(
        "Columns",
        len(df.columns),
    )


with col3:

    duplicates = (
        df.duplicated().sum()
    )

    st.metric(
        "Duplicate Rows",
        f"{duplicates:,}",
    )


with col4:

    st.metric(
        "Encoding",
        encoding,
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    "Preview Dataset",
    expanded=True,
):

    st.dataframe(
        df.head(100),
        width="stretch",
    )


# ============================================================
# COLUMN INFORMATION
# ============================================================

with st.expander(
    "Column Information"
):

    column_summary = (
        get_column_summary(df)
    )

    st.dataframe(
        column_summary,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# 2. TEXT COLUMN
# ============================================================

st.divider()

st.subheader(
    "2. Select Text Column"
)


text_candidates = (
    detect_text_columns(df)
)


if text_candidates:

    default_text_column = (
        text_candidates[0]
    )

else:

    default_text_column = (
        df.columns[0]
    )


default_index = (
    list(df.columns)
    .index(default_text_column)
)


text_column = st.selectbox(
    "Column containing text for topic modelling",
    options=df.columns.tolist(),
    index=default_index,
)


st.caption(
    f"Selected text column: **{text_column}**"
)


# ============================================================
# PREPARE TEXT
# ============================================================

working_df = prepare_text_data(
    df,
    text_column,
)


# ============================================================
# IMPORTANT:
# CREATE STABLE EMBEDDING INDEX
# ============================================================

working_df = (
    working_df
    .reset_index(drop=True)
)

working_df[
    "_embedding_index"
] = np.arange(
    len(working_df)
)


# ============================================================
# TEXT STATISTICS
# ============================================================

col1, col2, col3 = (
    st.columns(3)
)


with col1:

    st.metric(
        "Original Records",
        f"{len(df):,}",
    )


with col2:

    st.metric(
        "Valid Text Records",
        f"{len(working_df):,}",
    )


with col3:

    removed = (
        len(df)
        -
        len(working_df)
    )

    st.metric(
        "Empty Text Removed",
        f"{removed:,}",
    )


# ============================================================
# 3. DYNAMIC CATEGORY FILTERS
# ============================================================

st.divider()

st.subheader(
    "3. Dynamic Category Filters"
)

st.caption(
    "Add one or more filters. Multiple filters are combined "
    "using AND logic."
)


# ============================================================
# DETECT FILTER COLUMNS
# ============================================================

categorical_columns = (
    detect_categorical_columns(
        working_df,
        max_unique=200,
    )
)


categorical_columns = [
    c
    for c in categorical_columns
    if c not in [
        "_topic_text",
        "_embedding_index",
    ]
]


# ============================================================
# FILTER CONTROLS
# ============================================================

col_add, col_remove, col_reset = (
    st.columns(3)
)


with col_add:

    if st.button(
        "➕ Add Filter",
        width="stretch",
    ):

        st.session_state.filter_count += 1

        st.rerun()


with col_remove:

    if st.button(
        "➖ Remove Last Filter",
        width="stretch",
        disabled=(
            st.session_state.filter_count <= 1
        ),
    ):

        st.session_state.filter_count -= 1

        st.rerun()


with col_reset:

    if st.button(
        "🔄 Reset Filters",
        width="stretch",
    ):

        st.session_state.filter_count = 1

        keys_to_delete = [
            key
            for key in st.session_state.keys()
            if key.startswith(
                "dynamic_filter_"
            )
        ]

        for key in keys_to_delete:
            del st.session_state[key]

        st.rerun()


# ============================================================
# BUILD FILTERS
# ============================================================

selected_filters = {}

used_columns = []


for i in range(
    st.session_state.filter_count
):

    st.markdown(
        f"#### Filter {i + 1}"
    )

    available_columns = [
        c
        for c in categorical_columns
        if c not in used_columns
    ]

    if not available_columns:

        st.info(
            "All available categorical columns "
            "have already been selected."
        )

        break

    # --------------------------------------------------------
    # DEFAULT COLUMN
    # --------------------------------------------------------

    default_index = 0

    # First filter: relevance

    if i == 0:

        for j, column in enumerate(
            available_columns
        ):

            if "relevance" in column.lower():

                default_index = j
                break

    # Second filter: sentiment

    elif i == 1:

        for j, column in enumerate(
            available_columns
        ):

            if "sentiment" in column.lower():

                default_index = j
                break


    col1, col2 = st.columns(
        [1, 2]
    )


    # --------------------------------------------------------
    # SELECT COLUMN
    # --------------------------------------------------------

    with col1:

        selected_column = (
            st.selectbox(
                "Column",
                options=available_columns,
                index=default_index,
                key=f"dynamic_filter_column_{i}",
            )
        )


    used_columns.append(
        selected_column
    )


    # --------------------------------------------------------
    # SELECT VALUES
    # --------------------------------------------------------

    available_values = (
        get_filter_values(
            working_df,
            selected_column,
        )
    )


    with col2:

        selected_values = (
            st.multiselect(
                "Values",
                options=available_values,
                default=available_values,
                key=f"dynamic_filter_values_{i}",
            )
        )


    selected_filters[
        selected_column
    ] = selected_values


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = (
    working_df.copy()
)


for column, selected_values in (
    selected_filters.items()
):

    if selected_values:

        filtered_df = (
            filtered_df[
                filtered_df[column].isin(
                    selected_values
                )
            ]
            .copy()
        )


# ============================================================
# ACTIVE FILTER SUMMARY
# ============================================================

st.markdown(
    "#### Active Filter Summary"
)


if selected_filters:

    summary_data = []

    for column, values in (
        selected_filters.items()
    ):

        summary_data.append(
            {
                "Column": column,

                "Selected Values":
                    ", ".join(
                        map(str, values)
                    ),

                "Number Selected":
                    len(values),
            }
        )


    filter_summary_df = (
        pd.DataFrame(
            summary_data
        )
    )


    st.dataframe(
        filter_summary_df,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# 4. FILTERED CORPUS
# ============================================================

st.divider()

st.subheader(
    "4. Filtered Corpus"
)


col1, col2, col3 = (
    st.columns(3)
)


with col1:

    st.metric(
        "Available Texts",
        f"{len(working_df):,}",
    )


with col2:

    st.metric(
        "Selected Texts",
        f"{len(filtered_df):,}",
    )


with col3:

    if len(working_df) > 0:

        percentage = (
            len(filtered_df)
            /
            len(working_df)
            *
            100
        )

    else:

        percentage = 0


    st.metric(
        "Corpus Retained",
        f"{percentage:.1f}%",
    )


# ============================================================
# VALIDATE FILTERED DATA
# ============================================================

if len(filtered_df) == 0:

    st.error(
        "The current filters returned no documents."
    )

    st.stop()


elif len(filtered_df) < 50:

    st.warning(
        "Only a small number of documents remain. "
        "Topic modelling results may be unstable."
    )


# ============================================================
# PREVIEW FILTERED CORPUS
# ============================================================

with st.expander(
    "Preview Filtered Corpus",
    expanded=True,
):

    display_columns = [
        c
        for c in df.columns
        if c in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[
            display_columns
        ].head(200),

        width="stretch",
    )


# ============================================================
# 5. CATEGORY DISTRIBUTION
# ============================================================

st.divider()

st.subheader(
    "5. Category Distribution"
)


if categorical_columns:

    distribution_column = (
        st.selectbox(
            "View distribution for",
            options=categorical_columns,
        )
    )


    distribution = (
        filtered_df[
            distribution_column
        ]
        .value_counts(
            dropna=False
        )
        .reset_index()
    )


    distribution.columns = [
        distribution_column,
        "Count",
    ]


    col1, col2 = st.columns(
        [1, 2]
    )


    with col1:

        st.dataframe(
            distribution,
            width="stretch",
            hide_index=True,
        )


    with col2:

        st.bar_chart(
            distribution.set_index(
                distribution_column
            )
        )


# ============================================================
# DOWNLOAD FILTERED CORPUS
# ============================================================

download_filtered_df = (
    filtered_df.drop(
        columns=[
            "_embedding_index",
            "_topic_text",
        ],
        errors="ignore",
    )
)


csv_data = (
    download_filtered_df
    .to_csv(index=False)
    .encode("utf-8")
)


st.download_button(
    "Download Filtered Corpus",
    data=csv_data,
    file_name="filtered_topic_corpus.csv",
    mime="text/csv",
)


# ============================================================
# SAVE SESSION DATA
# ============================================================

st.session_state.filtered_df = (
    filtered_df
)

st.session_state.text_column = (
    text_column
)


# ============================================================
# 6. TOPIC MODELLING
# ============================================================

st.divider()

st.header(
    "6. Topic Modelling"
)


# ============================================================
# ANALYSIS MODE
# ============================================================

analysis_mode = st.radio(
    "Analysis Mode",
    [
        "Current Filter",
        "Overall Dataset",
        "Compare Categories",
    ],
    horizontal=True,
)


# ============================================================
# COMPARISON VARIABLES
# ============================================================

comparison_column_1 = None
comparison_column_2 = None

comparison_values_1 = []
comparison_values_2 = []


# ============================================================
# COMPARE CATEGORIES
# ============================================================

if analysis_mode == "Compare Categories":

    st.subheader(
        "Two-Dimensional Category Comparison"
    )

    st.info(
        "A single shared BERTopic model will be created. "
        "Topics will then be compared across two categories."
    )


    available_comparison_columns = (
        categorical_columns.copy()
    )


    if len(
        available_comparison_columns
    ) < 2:

        st.warning(
            "At least two categorical columns "
            "are required."
        )


    else:

        col1, col2 = (
            st.columns(2)
        )


        # ====================================================
        # DIMENSION 1
        # ====================================================

        with col1:

            default_1 = 0

            for i, column in enumerate(
                available_comparison_columns
            ):

                if (
                    "relevance"
                    in column.lower()
                ):

                    default_1 = i
                    break


            comparison_column_1 = (
                st.selectbox(
                    "Comparison Dimension 1",
                    options=(
                        available_comparison_columns
                    ),
                    index=default_1,
                    key="comparison_column_1",
                )
            )


        # ====================================================
        # DIMENSION 2
        # ====================================================

        second_options = [
            c
            for c in available_comparison_columns
            if c != comparison_column_1
        ]


        with col2:

            default_2 = 0

            for i, column in enumerate(
                second_options
            ):

                if (
                    "sentiment"
                    in column.lower()
                ):

                    default_2 = i
                    break


            comparison_column_2 = (
                st.selectbox(
                    "Comparison Dimension 2",
                    options=second_options,
                    index=default_2,
                    key="comparison_column_2",
                )
            )


        # ====================================================
        # VALUES
        # ====================================================

        st.markdown(
            "#### Categories to Include"
        )


        col1, col2 = (
            st.columns(2)
        )


        with col1:

            values_1 = (
                get_filter_values(
                    working_df,
                    comparison_column_1,
                )
            )


            comparison_values_1 = (
                st.multiselect(
                    comparison_column_1,
                    options=values_1,
                    default=values_1,
                    key="comparison_values_1",
                )
            )


        with col2:

            values_2 = (
                get_filter_values(
                    working_df,
                    comparison_column_2,
                )
            )


            comparison_values_2 = (
                st.multiselect(
                    comparison_column_2,
                    options=values_2,
                    default=values_2,
                    key="comparison_values_2",
                )
            )


        # ====================================================
        # COMBINATION PREVIEW
        # ====================================================

        comparison_preview = (
            working_df[
                working_df[
                    comparison_column_1
                ].isin(
                    comparison_values_1
                )
                &
                working_df[
                    comparison_column_2
                ].isin(
                    comparison_values_2
                )
            ]
            .copy()
        )


        combination_table = (
            comparison_preview
            .groupby(
                [
                    comparison_column_1,
                    comparison_column_2,
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="Documents"
            )
        )


        st.markdown(
            "#### Combination Preview"
        )


        st.dataframe(
            combination_table,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# BERTopic SETTINGS
# ============================================================

with st.expander(
    "BERTopic Settings",
    expanded=False,
):

    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:

        embedding_model_name = (
            st.selectbox(
                "Embedding Model",
                [
                    "all-MiniLM-L6-v2",
                    "all-mpnet-base-v2",
                    (
                        "paraphrase-multilingual-"
                        "MiniLM-L12-v2"
                    ),
                ],
                index=0,
            )
        )


    with col2:

        min_topic_size = (
            st.number_input(
                "Minimum Topic Size",
                min_value=5,
                max_value=500,
                value=15,
                step=5,
            )
        )


    with col3:

        top_n_words = (
            st.number_input(
                "Keywords per Topic",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
            )
        )


    col1, col2 = (
        st.columns(2)
    )


    with col1:

        ngram_min = (
            st.selectbox(
                "Minimum N-gram",
                [1, 2],
                index=0,
            )
        )


    with col2:

        ngram_max = (
            st.selectbox(
                "Maximum N-gram",
                [1, 2, 3],
                index=1,
            )
        )


# ============================================================
# SELECT MODELLING CORPUS
# ============================================================

if analysis_mode == "Current Filter":

    modelling_df = (
        filtered_df.copy()
    )


    st.info(
        f"BERTopic will analyse "
        f"{len(modelling_df):,} filtered documents."
    )


elif analysis_mode == "Overall Dataset":

    modelling_df = (
        working_df.copy()
    )


    st.info(
        f"BERTopic will analyse "
        f"{len(modelling_df):,} documents."
    )


else:

    if (
        comparison_column_1
        and comparison_column_2
        and comparison_values_1
        and comparison_values_2
    ):

        modelling_df = (
            working_df[
                working_df[
                    comparison_column_1
                ].isin(
                    comparison_values_1
                )
                &
                working_df[
                    comparison_column_2
                ].isin(
                    comparison_values_2
                )
            ]
            .copy()
        )

    else:

        modelling_df = (
            working_df.iloc[
                0:0
            ].copy()
        )


    st.info(
        f"Shared BERTopic model will analyse "
        f"{len(modelling_df):,} documents."
    )


# ============================================================
# LARGE DATA WARNING
# ============================================================

if len(modelling_df) > 20000:

    st.warning(
        f"You selected {len(modelling_df):,} documents. "
        "BERTopic may require considerable processing time "
        "on Streamlit Community Cloud. Embeddings will be "
        "cached after they are generated."
    )


# ============================================================
# RUN TOPIC MODEL
# ============================================================

run_model = st.button(
    "🚀 Run Topic Modelling",
    type="primary",
    width="stretch",
)


if run_model:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(modelling_df) < 20:

        st.error(
            "Too few documents for topic modelling."
        )

        st.stop()


    if ngram_min > ngram_max:

        st.error(
            "Minimum N-gram cannot be greater "
            "than Maximum N-gram."
        )

        st.stop()


    if min_topic_size > len(modelling_df):

        st.error(
            "Minimum Topic Size cannot be larger "
            "than the number of documents."
        )

        st.stop()


    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    with st.status(
        "Running BERTopic...",
        expanded=True,
    ) as status:

        try:

            # =================================================
            # STEP 1: INFORMATION
            # =================================================

            st.write(
                f"Full valid corpus: "
                f"{len(working_df):,} documents"
            )

            st.write(
                f"Documents selected for this analysis: "
                f"{len(modelling_df):,}"
            )

            st.write(
                f"Embedding model: "
                f"{embedding_model_name}"
            )


            # =================================================
            # STEP 2: PREPARE FULL CORPUS
            # =================================================

            all_documents = (
                working_df[
                    "_topic_text"
                ]
                .fillna("")
                .astype(str)
                .str.strip()
                .tolist()
            )


            # =================================================
            # STEP 3: EMBEDDINGS
            # =================================================

            st.write(
                "Loading cached embeddings or generating "
                "them for the first time..."
            )


            full_embeddings = (
                generate_cached_embeddings(
                    tuple(all_documents),
                    embedding_model_name,
                )
            )


            st.write(
                f"✓ Embeddings ready: "
                f"{full_embeddings.shape[0]:,} × "
                f"{full_embeddings.shape[1]:,}"
            )


            # =================================================
            # STEP 4: GET EMBEDDINGS FOR CURRENT SUBSET
            # =================================================

            embedding_indices = (
                modelling_df[
                    "_embedding_index"
                ]
                .astype(int)
                .to_numpy()
            )


            selected_embeddings = (
                full_embeddings[
                    embedding_indices
                ]
            )


            st.write(
                f"✓ Selected embeddings for "
                f"{len(selected_embeddings):,} documents."
            )


            # =================================================
            # STEP 5: BERTopic
            # =================================================

            st.write(
                "Running UMAP dimensionality reduction..."
            )

            st.write(
                "Running HDBSCAN clustering..."
            )

            st.write(
                "Generating topic representations..."
            )


            result = run_topic_model(
                modelling_df,
                text_column="_topic_text",
                embedding_model_name=(
                    embedding_model_name
                ),
                min_topic_size=(
                    min_topic_size
                ),
                nr_topics="auto",
                ngram_min=ngram_min,
                ngram_max=ngram_max,
                top_n_words=top_n_words,
                embeddings=selected_embeddings,
            )


            # =================================================
            # SAVE RESULTS
            # =================================================

            st.session_state.topic_result = (
                result
            )

            st.session_state.topic_analysis_mode = (
                analysis_mode
            )

            st.session_state[
                "topic_top_n_words"
            ] = top_n_words


            # =================================================
            # SAVE COMPARISON SETTINGS
            # =================================================

            if (
                analysis_mode
                ==
                "Compare Categories"
            ):

                st.session_state[
                    "comparison_column_1_used"
                ] = comparison_column_1

                st.session_state[
                    "comparison_column_2_used"
                ] = comparison_column_2


            # =================================================
            # COMPLETE
            # =================================================

            status.update(
                label=(
                    "Topic modelling completed."
                ),
                state="complete",
            )


        except Exception as e:

            status.update(
                label=(
                    "Topic modelling failed."
                ),
                state="error",
            )

            st.exception(e)


# ============================================================
# 7. TOPIC RESULTS
# ============================================================

if "topic_result" in st.session_state:

    result = (
        st.session_state.topic_result
    )


    topic_model = result[
        "model"
    ]

    result_df = result[
        "data"
    ]

    topic_info = result[
        "topic_info"
    ]


    stored_top_n_words = (
        st.session_state.get(
            "topic_top_n_words",
            20,
        )
    )


    st.divider()

    st.header(
        "7. Topic Results"
    )


    # ========================================================
    # METRICS
    # ========================================================

    valid_topics = (
        topic_info[
            topic_info["Topic"] != -1
        ]
    )


    number_topics = (
        len(valid_topics)
    )


    outlier_count = (
        result_df[
            "topic_id"
        ]
        .eq(-1)
        .sum()
    )


    if len(result_df) > 0:

        outlier_percentage = (
            outlier_count
            /
            len(result_df)
            *
            100
        )

    else:

        outlier_percentage = 0


    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:

        st.metric(
            "Topics Found",
            number_topics,
        )


    with col2:

        st.metric(
            "Outlier Documents",
            f"{outlier_count:,}",
        )


    with col3:

        st.metric(
            "Outlier Rate",
            f"{outlier_percentage:.1f}%",
        )


    # ========================================================
    # TOPIC TABLE
    # ========================================================

    st.subheader(
        "Topic Overview"
    )


    st.dataframe(
        topic_info,
        width="stretch",
        hide_index=True,
    )


    # ========================================================
    # TOPIC FREQUENCY
    # ========================================================

    st.subheader(
        "Topic Frequency"
    )


    frequency_data = (
        topic_info[
            topic_info[
                "Topic"
            ] != -1
        ][
            [
                "Topic",
                "Count",
            ]
        ]
        .set_index(
            "Topic"
        )
    )


    st.bar_chart(
        frequency_data
    )


    # ========================================================
    # TOPIC EXPLORER
    # ========================================================

    st.divider()

    st.subheader(
        "Topic Explorer"
    )


    available_topics = (
        topic_info[
            topic_info[
                "Topic"
            ] != -1
        ][
            "Topic"
        ]
        .tolist()
    )


    if available_topics:

        selected_topic = (
            st.selectbox(
                "Select Topic",
                options=available_topics,
            )
        )


        keywords = (
            get_topic_keywords(
                topic_model,
                selected_topic,
                top_n=stored_top_n_words,
            )
        )


        st.markdown(
            f"### Topic {selected_topic}"
        )


        # ====================================================
        # KEYWORDS
        # ====================================================

        keyword_df = (
            pd.DataFrame(
                keywords,
                columns=[
                    "Keyword",
                    "Score",
                ],
            )
        )


        col1, col2 = (
            st.columns(
                [1, 2]
            )
        )


        with col1:

            st.dataframe(
                keyword_df,
                width="stretch",
                hide_index=True,
            )


        with col2:

            if not keyword_df.empty:

                st.bar_chart(
                    keyword_df.set_index(
                        "Keyword"
                    )
                )


        # ====================================================
        # REPRESENTATIVE DOCUMENTS
        # ====================================================

        st.markdown(
            "### Representative Documents"
        )


        representative_docs = (
            get_representative_documents(
                topic_model,
                selected_topic,
            )
        )


        if representative_docs:

            for i, document in enumerate(
                representative_docs,
                start=1,
            ):

                with st.expander(
                    f"Representative Document {i}"
                ):

                    st.write(
                        document
                    )

        else:

            st.info(
                "No representative documents available."
            )


    # ========================================================
    # TOPIC VISUALIZATIONS
    # ========================================================

    st.divider()

    st.subheader(
        "Topic Visualizations"
    )


    visualization_type = (
        st.selectbox(
            "Select Visualization",
            [
                "Intertopic Distance Map (Bubble)",
                "Topic Word Scores",
                "Topic Hierarchy",
                "Topic Similarity Heatmap",
            ],
            key="topic_visualization_type",
        )
    )


    # ========================================================
    # 1. INTERTOPIC DISTANCE MAP
    # ========================================================

    if (
        visualization_type
        ==
        "Intertopic Distance Map (Bubble)"
    ):

        st.markdown(
            "### Intertopic Distance Map"
        )

        st.caption(
            "Each bubble represents a topic. "
            "Bubble size reflects topic frequency, while "
            "distance between bubbles indicates topic similarity."
        )

        try:

            max_topics_available = (
                len(
                    topic_info[
                        topic_info[
                            "Topic"
                        ] != -1
                    ]
                )
            )


            if max_topics_available > 0:

                number_topics_visual = (
                    st.slider(
                        "Number of topics to display",
                        min_value=1,
                        max_value=max_topics_available,
                        value=min(
                            30,
                            max_topics_available,
                        ),
                        step=1,
                    )
                )


                selected_visual_topics = (
                    topic_info[
                        topic_info[
                            "Topic"
                        ] != -1
                    ]
                    .sort_values(
                        "Count",
                        ascending=False,
                    )
                    .head(
                        number_topics_visual
                    )[
                        "Topic"
                    ]
                    .tolist()
                )


                fig = (
                    topic_model
                    .visualize_topics(
                        topics=(
                            selected_visual_topics
                        )
                    )
                )


                st.plotly_chart(
                    fig,
                    width="stretch",
                )

            else:

                st.info(
                    "No non-outlier topics are available "
                    "for visualization."
                )


        except Exception as e:

            st.warning(
                "Unable to generate the "
                "Intertopic Distance Map."
            )

            st.exception(e)


    # ========================================================
    # 2. TOPIC WORD SCORES
    # ========================================================

    elif (
        visualization_type
        ==
        "Topic Word Scores"
    ):

        st.markdown(
            "### Topic Keyword Scores"
        )

        try:

            fig = (
                topic_model
                .visualize_barchart(
                    top_n_topics=min(
                        20,
                        max(
                            1,
                            number_topics,
                        ),
                    ),
                    n_words=min(
                        20,
                        stored_top_n_words,
                    ),
                )
            )


            st.plotly_chart(
                fig,
                width="stretch",
            )


        except Exception as e:

            st.warning(
                "Unable to generate topic "
                "word visualization."
            )

            st.exception(e)


    # ========================================================
    # 3. TOPIC HIERARCHY
    # ========================================================

    elif (
        visualization_type
        ==
        "Topic Hierarchy"
    ):

        st.markdown(
            "### Topic Hierarchy"
        )

        st.caption(
            "This visualization shows how similar topics "
            "can be grouped into broader themes."
        )

        try:

            fig = (
                topic_model
                .visualize_hierarchy()
            )


            st.plotly_chart(
                fig,
                width="stretch",
            )


        except Exception as e:

            st.warning(
                "Unable to generate topic hierarchy."
            )

            st.exception(e)


    # ========================================================
    # 4. TOPIC SIMILARITY HEATMAP
    # ========================================================

    elif (
        visualization_type
        ==
        "Topic Similarity Heatmap"
    ):

        st.markdown(
            "### Topic Similarity Heatmap"
        )

        try:

            fig = (
                topic_model
                .visualize_heatmap()
            )


            st.plotly_chart(
                fig,
                width="stretch",
            )


        except Exception as e:

            st.warning(
                "Unable to generate topic "
                "similarity heatmap."
            )

            st.exception(e)


    # ========================================================
    # DOWNLOAD TOPIC RESULTS
    # ========================================================

    st.divider()

    download_result_df = (
        result_df.drop(
            columns=[
                "_embedding_index",
                "_topic_text",
            ],
            errors="ignore",
        )
    )


    topic_csv = (
        download_result_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(
        "Download Dataset with Topic IDs",
        data=topic_csv,
        file_name=(
            "topic_modelling_results.csv"
        ),
        mime="text/csv",
    )


# ============================================================
# 8. TWO-DIMENSIONAL COMPARISON
# ============================================================

if (
    "topic_result"
    in st.session_state
    and
    st.session_state.get(
        "topic_analysis_mode"
    )
    ==
    "Compare Categories"
):

    st.divider()

    st.header(
        "8. Topic × Category Analysis"
    )


    comparison_df = (
        st.session_state[
            "topic_result"
        ][
            "data"
        ]
        .copy()
    )


    comparison_col_1 = (
        st.session_state.get(
            "comparison_column_1_used"
        )
    )


    comparison_col_2 = (
        st.session_state.get(
            "comparison_column_2_used"
        )
    )


    if (
        comparison_col_1
        and
        comparison_col_2
    ):

        st.write(
            f"Comparing **{comparison_col_1}** "
            f"× **{comparison_col_2}**"
        )


        # ====================================================
        # OUTLIERS
        # ====================================================

        exclude_outliers = (
            st.checkbox(
                "Exclude BERTopic outliers (Topic -1)",
                value=True,
            )
        )


        analysis_df = (
            comparison_df.copy()
        )


        if exclude_outliers:

            analysis_df = (
                analysis_df[
                    analysis_df[
                        "topic_id"
                    ] != -1
                ]
            )


        # ====================================================
        # GROUP TOPICS
        # ====================================================

        topic_category = (
            analysis_df
            .groupby(
                [
                    "topic_id",
                    comparison_col_1,
                    comparison_col_2,
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="Documents"
            )
        )


        # ====================================================
        # CATEGORY TOTALS
        # ====================================================

        category_totals = (
            topic_category
            .groupby(
                [
                    comparison_col_1,
                    comparison_col_2,
                ]
            )[
                "Documents"
            ]
            .transform(
                "sum"
            )
        )


        topic_category[
            "Percentage"
        ] = (
            topic_category[
                "Documents"
            ]
            /
            category_totals
            *
            100
        ).round(2)


        # ====================================================
        # COMPLETE TABLE
        # ====================================================

        st.subheader(
            "Complete Topic × Category Table"
        )


        st.dataframe(
            topic_category,
            width="stretch",
            hide_index=True,
        )


        # ====================================================
        # SELECT TOPIC
        # ====================================================

        comparison_topics = sorted(
            analysis_df[
                "topic_id"
            ].unique()
        )


        if comparison_topics:

            selected_comparison_topic = (
                st.selectbox(
                    "Select Topic to Compare",
                    options=(
                        comparison_topics
                    ),
                    key=(
                        "comparison_topic_selector"
                    ),
                )
            )


            selected_topic_df = (
                topic_category[
                    topic_category[
                        "topic_id"
                    ]
                    ==
                    selected_comparison_topic
                ]
                .copy()
            )


            # ================================================
            # COUNT MATRIX
            # ================================================

            count_matrix = (
                selected_topic_df
                .pivot_table(
                    index=comparison_col_1,
                    columns=comparison_col_2,
                    values="Documents",
                    fill_value=0,
                )
            )


            st.markdown(
                f"### Topic "
                f"{selected_comparison_topic}"
            )


            st.markdown(
                "#### Number of Documents"
            )


            st.dataframe(
                count_matrix,
                width="stretch",
            )


            # ================================================
            # COUNT HEATMAP
            # ================================================

            fig_count = px.imshow(
                count_matrix,
                text_auto=True,
                aspect="auto",
                labels={
                    "x": comparison_col_2,
                    "y": comparison_col_1,
                    "color": "Documents",
                },
                title=(
                    f"Topic "
                    f"{selected_comparison_topic}: "
                    f"{comparison_col_1} × "
                    f"{comparison_col_2}"
                ),
            )


            st.plotly_chart(
                fig_count,
                width="stretch",
            )


            # ================================================
            # PERCENTAGE MATRIX
            # ================================================

            percentage_matrix = (
                selected_topic_df
                .pivot_table(
                    index=comparison_col_1,
                    columns=comparison_col_2,
                    values="Percentage",
                    fill_value=0,
                )
            )


            st.markdown(
                "#### Topic Prevalence (%)"
            )


            st.dataframe(
                percentage_matrix,
                width="stretch",
            )


            # ================================================
            # PERCENTAGE HEATMAP
            # ================================================

            fig_percentage = (
                px.imshow(
                    percentage_matrix,
                    text_auto=".1f",
                    aspect="auto",
                    labels={
                        "x": comparison_col_2,
                        "y": comparison_col_1,
                        "color": "%",
                    },
                    title=(
                        "Topic prevalence within "
                        "each category (%)"
                    ),
                )
            )


            st.plotly_chart(
                fig_percentage,
                width="stretch",
            )


            # ================================================
            # BAR CHART
            # ================================================

            selected_topic_df[
                "Category"
            ] = (
                selected_topic_df[
                    comparison_col_1
                ].astype(str)
                +
                " | "
                +
                selected_topic_df[
                    comparison_col_2
                ].astype(str)
            )


            bar_fig = px.bar(
                selected_topic_df,
                x="Category",
                y="Documents",
                text="Documents",
                title=(
                    f"Topic "
                    f"{selected_comparison_topic} "
                    "Distribution"
                ),
            )


            st.plotly_chart(
                bar_fig,
                width="stretch",
            )


        # ====================================================
        # DOWNLOAD COMPARISON
        # ====================================================

        comparison_csv = (
            topic_category
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )


        st.download_button(
            "Download Category Comparison",
            data=comparison_csv,
            file_name=(
                "topic_category_comparison.csv"
            ),
            mime="text/csv",
        )


    else:

        st.warning(
            "Comparison columns are unavailable. "
            "Run Compare Categories again."
        )
