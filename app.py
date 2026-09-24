# ============================================================
# DYNAMIC TOPIC MODELLING SYSTEM
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


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
# CACHE EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_cached_embedding_model(
    model_name
):

    return load_embedding_model(
        model_name
    )


# ============================================================
# CACHE EMBEDDINGS
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=3,
)
def generate_cached_embeddings(
    documents,
    model_name,
):

    model = (
        get_cached_embedding_model(
            model_name
        )
    )


    return create_embeddings(
        documents=list(documents),
        embedding_model=model,
        batch_size=64,
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "Dynamic Topic Modelling System"
)

st.caption(
    "Explore topics dynamically across relevance, sentiment, "
    "emotion, source, time and other dataset categories."
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "filter_count" not in st.session_state:

    st.session_state.filter_count = 1


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Dataset"
    )

    uploaded_file = (
        st.file_uploader(
            "Upload CSV file",
            type=["csv"],
        )
    )


# ============================================================
# CHECK FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a CSV file from the sidebar to begin."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df, encoding = (
        load_csv(
            uploaded_file
        )
    )

except Exception as e:

    st.error(
        f"Unable to load dataset:\n\n{e}"
    )

    st.stop()


# ============================================================
# 1. DATASET OVERVIEW
# ============================================================

st.subheader(
    "1. Dataset Overview"
)


c1, c2, c3, c4 = (
    st.columns(4)
)


with c1:

    st.metric(
        "Rows",
        f"{len(df):,}",
    )


with c2:

    st.metric(
        "Columns",
        len(df.columns),
    )


with c3:

    st.metric(
        "Duplicate Rows",
        f"{df.duplicated().sum():,}",
    )


with c4:

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
# 2. SELECT TEXT COLUMN
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
    .index(
        default_text_column
    )
)


text_column = (
    st.selectbox(
        "Column containing text for topic modelling",
        options=df.columns.tolist(),
        index=default_index,
    )
)


# ============================================================
# PREPARE TEXT
# ============================================================

working_df = (
    prepare_text_data(
        df,
        text_column,
    )
)


working_df = (
    working_df
    .reset_index(drop=True)
)


# Stable index for embedding lookup

working_df[
    "_embedding_index"
] = np.arange(
    len(working_df)
)


# ============================================================
# TEXT STATISTICS
# ============================================================

c1, c2, c3 = (
    st.columns(3)
)


with c1:

    st.metric(
        "Original Records",
        f"{len(df):,}",
    )


with c2:

    st.metric(
        "Valid Text Records",
        f"{len(working_df):,}",
    )


with c3:

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
# 3. DYNAMIC FILTERS
# ============================================================

st.divider()

st.subheader(
    "3. Dynamic Category Filters"
)

st.caption(
    "Multiple filters are combined using AND logic."
)


categorical_columns = (
    detect_categorical_columns(
        working_df,
        max_unique=200,
    )
)


categorical_columns = [

    column

    for column
    in categorical_columns

    if column not in [
        "_topic_text",
        "_embedding_index",
    ]
]


# ============================================================
# FILTER BUTTONS
# ============================================================

c1, c2, c3 = (
    st.columns(3)
)


with c1:

    if st.button(
        "➕ Add Filter",
        width="stretch",
    ):

        st.session_state.filter_count += 1

        st.rerun()


with c2:

    if st.button(
        "➖ Remove Last Filter",
        width="stretch",
        disabled=(
            st.session_state.filter_count
            <= 1
        ),
    ):

        st.session_state.filter_count -= 1

        st.rerun()


with c3:

    if st.button(
        "🔄 Reset Filters",
        width="stretch",
    ):

        st.session_state.filter_count = 1

        for key in list(
            st.session_state.keys()
        ):

            if key.startswith(
                "dynamic_filter_"
            ):

                del st.session_state[
                    key
                ]

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

        column

        for column
        in categorical_columns

        if column not in used_columns
    ]


    if not available_columns:

        st.info(
            "All available categorical columns "
            "have been selected."
        )

        break


    default_filter_index = 0


    # First filter -> relevance

    if i == 0:

        for j, column in enumerate(
            available_columns
        ):

            if "relevance" in column.lower():

                default_filter_index = j

                break


    # Second filter -> sentiment

    elif i == 1:

        for j, column in enumerate(
            available_columns
        ):

            if "sentiment" in column.lower():

                default_filter_index = j

                break


    c1, c2 = (
        st.columns(
            [1, 2]
        )
    )


    with c1:

        selected_column = (
            st.selectbox(
                "Column",
                options=available_columns,
                index=default_filter_index,
                key=(
                    f"dynamic_filter_column_{i}"
                ),
            )
        )


    used_columns.append(
        selected_column
    )


    available_values = (
        get_filter_values(
            working_df,
            selected_column,
        )
    )


    with c2:

        selected_values = (
            st.multiselect(
                "Values",
                options=available_values,
                default=available_values,
                key=(
                    f"dynamic_filter_values_{i}"
                ),
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


for column, values in (
    selected_filters.items()
):

    if values:

        filtered_df = (
            filtered_df[
                filtered_df[
                    column
                ].isin(values)
            ]
            .copy()
        )


# ============================================================
# ACTIVE FILTER SUMMARY
# ============================================================

st.markdown(
    "#### Active Filter Summary"
)


summary_rows = []


for column, values in (
    selected_filters.items()
):

    summary_rows.append(
        {
            "Column":
                column,

            "Selected Values":
                ", ".join(
                    map(
                        str,
                        values,
                    )
                ),

            "Number Selected":
                len(values),
        }
    )


if summary_rows:

    st.dataframe(
        pd.DataFrame(
            summary_rows
        ),
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


c1, c2, c3 = (
    st.columns(3)
)


with c1:

    st.metric(
        "Available Texts",
        f"{len(working_df):,}",
    )


with c2:

    st.metric(
        "Selected Texts",
        f"{len(filtered_df):,}",
    )


with c3:

    percentage = (

        (
            len(filtered_df)
            /
            len(working_df)
            *
            100
        )

        if len(working_df) > 0

        else 0
    )


    st.metric(
        "Corpus Retained",
        f"{percentage:.1f}%",
    )


if len(filtered_df) == 0:

    st.error(
        "Current filters returned no documents."
    )

    st.stop()


if len(filtered_df) < 50:

    st.warning(
        "Only a small number of documents remain. "
        "Topic modelling may be unstable."
    )


# ============================================================
# PREVIEW
# ============================================================

with st.expander(
    "Preview Filtered Corpus"
):

    preview_columns = [

        column

        for column
        in df.columns

        if column in filtered_df.columns
    ]


    st.dataframe(
        filtered_df[
            preview_columns
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
            categorical_columns,
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


    c1, c2 = (
        st.columns(
            [1, 2]
        )
    )


    with c1:

        st.dataframe(
            distribution,
            width="stretch",
            hide_index=True,
        )


    with c2:

        st.bar_chart(
            distribution.set_index(
                distribution_column
            )
        )


# ============================================================
# DOWNLOAD FILTERED CORPUS
# ============================================================

filtered_download = (
    filtered_df.drop(
        columns=[
            "_topic_text",
            "_embedding_index",
        ],
        errors="ignore",
    )
)


st.download_button(
    "⬇️ Download Filtered Corpus",
    data=(
        filtered_download
        .to_csv(index=False)
        .encode("utf-8-sig")
    ),
    file_name=(
        "filtered_topic_corpus.csv"
    ),
    mime="text/csv",
)


# ============================================================
# 6. TOPIC MODELLING
# ============================================================

st.divider()

st.header(
    "6. Topic Modelling"
)


analysis_mode = (
    st.radio(
        "Analysis Mode",
        [
            "Current Filter",
            "Overall Dataset",
            "Compare Categories",
        ],
        horizontal=True,
    )
)


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


    if len(
        categorical_columns
    ) >= 2:

        c1, c2 = (
            st.columns(2)
        )


        # ----------------------------------------------------
        # First dimension
        # ----------------------------------------------------

        default_1 = 0


        for i, column in enumerate(
            categorical_columns
        ):

            if "relevance" in column.lower():

                default_1 = i

                break


        with c1:

            comparison_column_1 = (
                st.selectbox(
                    "Comparison Dimension 1",
                    categorical_columns,
                    index=default_1,
                    key="comparison_column_1",
                )
            )


        # ----------------------------------------------------
        # Second dimension
        # ----------------------------------------------------

        second_options = [

            column

            for column
            in categorical_columns

            if column
            != comparison_column_1
        ]


        default_2 = 0


        for i, column in enumerate(
            second_options
        ):

            if "sentiment" in column.lower():

                default_2 = i

                break


        with c2:

            comparison_column_2 = (
                st.selectbox(
                    "Comparison Dimension 2",
                    second_options,
                    index=default_2,
                    key="comparison_column_2",
                )
            )


        c1, c2 = (
            st.columns(2)
        )


        with c1:

            values_1 = (
                get_filter_values(
                    working_df,
                    comparison_column_1,
                )
            )


            comparison_values_1 = (
                st.multiselect(
                    comparison_column_1,
                    values_1,
                    default=values_1,
                    key="comparison_values_1",
                )
            )


        with c2:

            values_2 = (
                get_filter_values(
                    working_df,
                    comparison_column_2,
                )
            )


            comparison_values_2 = (
                st.multiselect(
                    comparison_column_2,
                    values_2,
                    default=values_2,
                    key="comparison_values_2",
                )
            )


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


        st.dataframe(
            combination_table,
            width="stretch",
            hide_index=True,
        )


    else:

        st.warning(
            "At least two categorical columns are required."
        )


# ============================================================
# BERTopic SETTINGS
# ============================================================

with st.expander(
    "BERTopic Settings"
):

    c1, c2, c3 = (
        st.columns(3)
    )


    with c1:

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


    with c2:

        min_topic_size = (
            st.number_input(
                "Minimum Topic Size",
                min_value=5,
                max_value=500,
                value=15,
                step=5,
            )
        )


    with c3:

        top_n_words = (
            st.number_input(
                "Keywords per Topic",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
            )
        )


    c1, c2 = (
        st.columns(2)
    )


    with c1:

        ngram_min = (
            st.selectbox(
                "Minimum N-gram",
                [1, 2],
                index=0,
            )
        )


    with c2:

        ngram_max = (
            st.selectbox(
                "Maximum N-gram",
                [1, 2, 3],
                index=1,
            )
        )


# ============================================================
# SELECT MODELLING DATA
# ============================================================

if analysis_mode == "Current Filter":

    modelling_df = (
        filtered_df.copy()
    )


elif analysis_mode == "Overall Dataset":

    modelling_df = (
        working_df.copy()
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
    f"BERTopic will analyse "
    f"{len(modelling_df):,} documents."
)


if len(modelling_df) > 20000:

    st.warning(
        "Large corpus selected. The first run may take "
        "considerable time because embeddings must be generated. "
        "Subsequent runs can reuse cached embeddings."
    )


# ============================================================
# RUN MODEL
# ============================================================

run_model = (
    st.button(
        "🚀 Run Topic Modelling",
        type="primary",
        width="stretch",
    )
)


if run_model:

    if len(modelling_df) < 20:

        st.error(
            "Too few documents for topic modelling."
        )

        st.stop()


    if ngram_min > ngram_max:

        st.error(
            "Minimum N-gram cannot exceed Maximum N-gram."
        )

        st.stop()


    with st.status(
        "Running BERTopic...",
        expanded=True,
    ) as status:

        try:

            # =================================================
            # FULL CORPUS DOCUMENTS
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


            st.write(
                f"Full valid corpus: "
                f"{len(all_documents):,} documents"
            )


            st.write(
                f"Current analysis: "
                f"{len(modelling_df):,} documents"
            )


            st.write(
                f"Embedding model: "
                f"{embedding_model_name}"
            )


            # =================================================
            # EMBEDDINGS
            # =================================================

            st.write(
                "Loading cached embeddings or generating "
                "embeddings for the first run..."
            )


            full_embeddings = (
                generate_cached_embeddings(
                    tuple(all_documents),
                    embedding_model_name,
                )
            )


            st.write(
                f"✓ Full embedding matrix ready: "
                f"{full_embeddings.shape}"
            )


            # =================================================
            # SELECT REQUIRED EMBEDDINGS
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
                f"✓ Selected "
                f"{len(selected_embeddings):,} embeddings"
            )


            # =================================================
            # BERTopic
            # =================================================

            st.write(
                "Running UMAP and HDBSCAN..."
            )


            result = (
                run_topic_model(
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
                    embeddings=(
                        selected_embeddings
                    ),
                )
            )


            st.session_state[
                "topic_result"
            ] = result


            st.session_state[
                "topic_analysis_mode"
            ] = analysis_mode


            st.session_state[
                "topic_top_n_words"
            ] = top_n_words


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
        st.session_state[
            "topic_result"
        ]
    )


    topic_model = (
        result["model"]
    )

    result_df = (
        result["data"]
    )

    topic_info = (
        result["topic_info"]
    )


    top_words = (
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
            topic_info[
                "Topic"
            ] != -1
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


    outlier_percentage = (

        outlier_count
        /
        len(result_df)
        *
        100

        if len(result_df)

        else 0
    )


    c1, c2, c3 = (
        st.columns(3)
    )


    with c1:

        st.metric(
            "Topics Found",
            number_topics,
        )


    with c2:

        st.metric(
            "Outlier Documents",
            f"{outlier_count:,}",
        )


    with c3:

        st.metric(
            "Outlier Rate",
            f"{outlier_percentage:.1f}%",
        )


    # ========================================================
    # TOPIC OVERVIEW
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
        valid_topics[
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
        valid_topics[
            "Topic"
        ]
        .tolist()
    )


    if available_topics:

        selected_topic = (
            st.selectbox(
                "Select Topic",
                available_topics,
            )
        )


        # ----------------------------------------------------
        # Auto label
        # ----------------------------------------------------

        selected_info = (
            topic_info[
                topic_info[
                    "Topic"
                ]
                ==
                selected_topic
            ]
        )


        if not selected_info.empty:

            auto_label = (
                selected_info[
                    "Auto_Label"
                ]
                .iloc[0]
            )


            st.markdown(
                f"### Topic {selected_topic}"
            )

            st.markdown(
                f"**Automatic Label:** "
                f"{auto_label}"
            )


        # ----------------------------------------------------
        # Keywords
        # ----------------------------------------------------

        keywords = (
            get_topic_keywords(
                topic_model,
                selected_topic,
                top_n=top_words,
            )
        )


        keyword_df = (
            pd.DataFrame(
                keywords,
                columns=[
                    "Keyword",
                    "Score",
                ],
            )
        )


        c1, c2 = (
            st.columns(
                [1, 2]
            )
        )


        with c1:

            st.dataframe(
                keyword_df,
                width="stretch",
                hide_index=True,
            )


        with c2:

            if not keyword_df.empty:

                st.bar_chart(
                    keyword_df
                    .set_index(
                        "Keyword"
                    )
                )


        # ----------------------------------------------------
        # Representative docs
        # ----------------------------------------------------

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
                1,
            ):

                with st.expander(
                    f"Representative Document {i}"
                ):

                    st.write(
                        document
                    )


    # ========================================================
    # VISUALIZATIONS
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
        )
    )


    try:

        if (
            visualization_type
            ==
            "Intertopic Distance Map (Bubble)"
        ):

            if number_topics > 0:

                max_display = (
                    st.slider(
                        "Number of topics to display",
                        1,
                        number_topics,
                        min(
                            30,
                            number_topics,
                        ),
                    )
                )


                selected_visual_topics = (
                    valid_topics
                    .sort_values(
                        "Count",
                        ascending=False,
                    )
                    .head(
                        max_display
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


        elif (
            visualization_type
            ==
            "Topic Word Scores"
        ):

            fig = (
                topic_model
                .visualize_barchart(
                    top_n_topics=min(
                        20,
                        number_topics,
                    ),
                    n_words=min(
                        20,
                        top_words,
                    ),
                )
            )


            st.plotly_chart(
                fig,
                width="stretch",
            )


        elif (
            visualization_type
            ==
            "Topic Hierarchy"
        ):

            fig = (
                topic_model
                .visualize_hierarchy()
            )


            st.plotly_chart(
                fig,
                width="stretch",
            )


        elif (
            visualization_type
            ==
            "Topic Similarity Heatmap"
        ):

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
            "Unable to generate this visualization."
        )

        st.exception(e)


    # ========================================================
    # 8. DOWNLOAD RESULTS
    # ========================================================

    st.divider()

    st.header(
        "8. Download Results"
    )


    # ========================================================
    # DOCUMENT LEVEL
    # ========================================================

    st.subheader(
        "Document-Level Results"
    )


    st.caption(
        "Every document is assigned its BERTopic topic ID, "
        "automatic topic label and topic keywords."
    )


    document_output = (
        result_df.drop(
            columns=[
                "_topic_text",
                "_embedding_index",
            ],
            errors="ignore",
        )
        .copy()
    )


    # Put topic columns together at end

    topic_columns = [
        "topic_id",
        "topic_auto_label",
        "topic_keywords",
    ]


    topic_columns = [

        column

        for column
        in topic_columns

        if column
        in document_output.columns
    ]


    other_columns = [

        column

        for column
        in document_output.columns

        if column
        not in topic_columns
    ]


    document_output = (
        document_output[
            other_columns
            +
            topic_columns
        ]
    )


    with st.expander(
        "Preview Document-Level Results"
    ):

        st.dataframe(
            document_output.head(100),
            width="stretch",
        )


    document_csv = (
        document_output
        .to_csv(
            index=False
        )
        .encode(
            "utf-8-sig"
        )
    )


    st.download_button(
        "⬇️ Download Document-Level Topic Results",
        data=document_csv,
        file_name=(
            "topic_modelling_document_results.csv"
        ),
        mime="text/csv",
        width="stretch",
    )


    # ========================================================
    # TOPIC SUMMARY
    # ========================================================

    st.subheader(
        "Topic-Level Summary"
    )


    st.caption(
        "One row per topic with frequency, automatic label "
        "and complete topic keyword representation."
    )


    preferred_columns = [
        "Topic",
        "Count",
        "Auto_Label",
        "Topic_Keywords",
        "Name",
        "Representation",
        "Representative_Docs",
        "Topic_Status",
    ]


    available_summary_columns = [

        column

        for column
        in preferred_columns

        if column
        in topic_info.columns
    ]


    topic_summary = (
        topic_info[
            available_summary_columns
        ]
        .copy()
    )


    st.dataframe(
        topic_summary,
        width="stretch",
        hide_index=True,
    )


    topic_summary_csv = (
        topic_summary
        .to_csv(
            index=False
        )
        .encode(
            "utf-8-sig"
        )
    )


    st.download_button(
        "⬇️ Download Topic Summary",
        data=topic_summary_csv,
        file_name="topic_summary.csv",
        mime="text/csv",
        width="stretch",
    )


# ============================================================
# 9. TWO-DIMENSIONAL CATEGORY ANALYSIS
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
        and comparison_col_2
    ):

        st.divider()

        st.header(
            "9. Topic × Category Analysis"
        )


        st.write(
            f"Comparing "
            f"**{comparison_col_1}** × "
            f"**{comparison_col_2}**"
        )


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
        # GROUP
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
        # ADD TOPIC LABEL
        # ====================================================

        label_map = (

            comparison_df[
                [
                    "topic_id",
                    "topic_auto_label",
                ]
            ]

            .drop_duplicates(
                subset=[
                    "topic_id"
                ]
            )

            .set_index(
                "topic_id"
            )[
                "topic_auto_label"
            ]

            .to_dict()
        )


        topic_category[
            "Topic_Label"
        ] = (
            topic_category[
                "topic_id"
            ]
            .map(
                label_map
            )
        )


        # ====================================================
        # PERCENTAGE
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
        # TABLE
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
            ]
            .unique()
        )


        if comparison_topics:

            selected_comparison_topic = (
                st.selectbox(
                    "Select Topic to Compare",
                    comparison_topics,
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


            selected_label = (
                label_map.get(
                    selected_comparison_topic,
                    "",
                )
            )


            st.markdown(
                f"### Topic "
                f"{selected_comparison_topic}"
            )


            st.markdown(
                f"**{selected_label}**"
            )


            # =================================================
            # COUNT MATRIX
            # =================================================

            count_matrix = (
                selected_topic_df
                .pivot_table(
                    index=(
                        comparison_col_1
                    ),
                    columns=(
                        comparison_col_2
                    ),
                    values="Documents",
                    fill_value=0,
                )
            )


            st.markdown(
                "#### Number of Documents"
            )


            st.dataframe(
                count_matrix,
                width="stretch",
            )


            fig_count = (
                px.imshow(
                    count_matrix,
                    text_auto=True,
                    aspect="auto",
                    labels={
                        "x":
                            comparison_col_2,

                        "y":
                            comparison_col_1,

                        "color":
                            "Documents",
                    },
                    title=(
                        f"Topic "
                        f"{selected_comparison_topic}: "
                        f"{selected_label}"
                    ),
                )
            )


            st.plotly_chart(
                fig_count,
                width="stretch",
            )


            # =================================================
            # PERCENTAGE MATRIX
            # =================================================

            percentage_matrix = (
                selected_topic_df
                .pivot_table(
                    index=(
                        comparison_col_1
                    ),
                    columns=(
                        comparison_col_2
                    ),
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


            fig_percentage = (
                px.imshow(
                    percentage_matrix,
                    text_auto=".1f",
                    aspect="auto",
                    labels={
                        "x":
                            comparison_col_2,

                        "y":
                            comparison_col_1,

                        "color":
                            "%",
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


            # =================================================
            # BAR CHART
            # =================================================

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


            bar_fig = (
                px.bar(
                    selected_topic_df,
                    x="Category",
                    y="Documents",
                    text="Documents",
                    title=(
                        f"Topic "
                        f"{selected_comparison_topic}: "
                        f"{selected_label}"
                    ),
                )
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
                "utf-8-sig"
            )
        )


        st.download_button(
            "⬇️ Download Category Comparison",
            data=comparison_csv,
            file_name=(
                "topic_category_comparison.csv"
            ),
            mime="text/csv",
            width="stretch",
        )
