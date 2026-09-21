import streamlit as st
import pandas as pd
import plotly.express as px

from modules.data_loader import (
    load_csv,
    get_column_summary,
    detect_text_columns,
    detect_categorical_columns,
)

from modules.filters import (
    apply_category_filters,
    prepare_text_data,
    get_filter_values,
)

from modules.topic_model import (
    run_topic_model,
    get_topic_keywords,
    get_representative_documents,
)




# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dynamic Topic Modelling",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("Dynamic Topic Modelling System")

st.caption(
    "Explore topics dynamically across categories, "
    "sentiment, emotion, source, time, and other variables."
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "data" not in st.session_state:
    st.session_state.data = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
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

    df, encoding = load_csv(uploaded_file)

except Exception as e:

    st.error(
        f"Unable to load the dataset:\n\n{e}"
    )

    st.stop()


st.session_state.data = df


# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader("1. Dataset Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Rows",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "Columns",
        len(df.columns)
    )


with col3:

    duplicates = df.duplicated().sum()

    st.metric(
        "Duplicate Rows",
        f"{duplicates:,}"
    )


with col4:

    st.metric(
        "Encoding",
        encoding
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    "Preview Dataset",
    expanded=True
):

    st.dataframe(
        df.head(100),
        use_container_width=True
    )


# ============================================================
# COLUMN INFORMATION
# ============================================================

with st.expander(
    "Column Information"
):

    column_summary = get_column_summary(df)

    st.dataframe(
        column_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TEXT COLUMN
# ============================================================

st.divider()

st.subheader("2. Select Text Column")


text_candidates = detect_text_columns(df)


if text_candidates:

    default_text_column = text_candidates[0]

else:

    default_text_column = df.columns[0]


default_index = list(df.columns).index(
    default_text_column
)


text_column = st.selectbox(
    "Column containing the text for topic modelling",
    options=df.columns.tolist(),
    index=default_index
)


st.caption(
    f"Selected text column: **{text_column}**"
)


# ============================================================
# PREPARE TEXT
# ============================================================

working_df = prepare_text_data(
    df,
    text_column
)


# ============================================================
# TEXT STATISTICS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Original Records",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "Valid Text Records",
        f"{len(working_df):,}"
    )


with col3:

    removed = len(df) - len(working_df)

    st.metric(
        "Empty Text Removed",
        f"{removed:,}"
    )


# ============================================================
# DYNAMIC FILTERS
# ============================================================

st.divider()

st.subheader("3. Dynamic Category Filters")


categorical_columns = detect_categorical_columns(
    working_df,
    max_unique=100
)


# Do not use internal topic text as filter
categorical_columns = [
    c
    for c in categorical_columns
    if c != "_topic_text"
]


if len(categorical_columns) == 0:

    st.warning(
        "No suitable categorical columns were automatically detected."
    )

    selected_filter_columns = []

else:

    selected_filter_columns = st.multiselect(
        "Select columns to use as filters",
        options=categorical_columns,
        help=(
            "Examples: sentiment, emotion, source, "
            "subreddit, year, gender, label."
        )
    )


# ============================================================
# CREATE FILTERS
# ============================================================

selected_filters = {}


if selected_filter_columns:

    st.markdown("#### Filter Values")

    for column in selected_filter_columns:

        values = get_filter_values(
            working_df,
            column
        )

        selected_values = st.multiselect(
            f"{column}",
            options=values,
            default=values,
            key=f"filter_{column}"
        )

        selected_filters[column] = selected_values


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = apply_category_filters(
    working_df,
    selected_filters
)


# ============================================================
# FILTER RESULTS
# ============================================================

st.divider()

st.subheader("4. Filtered Corpus")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Available Texts",
        f"{len(working_df):,}"
    )


with col2:

    st.metric(
        "Selected Texts",
        f"{len(filtered_df):,}"
    )


with col3:

    if len(working_df) > 0:

        percentage = (
            len(filtered_df)
            / len(working_df)
        ) * 100

    else:

        percentage = 0

    st.metric(
        "Corpus Retained",
        f"{percentage:.1f}%"
    )


# ============================================================
# WARNING
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
# FILTER SUMMARY
# ============================================================

if selected_filters:

    with st.expander(
        "Current Filter Configuration"
    ):

        for column, values in selected_filters.items():

            st.write(
                f"**{column}:** "
                + ", ".join(map(str, values))
            )


# ============================================================
# PREVIEW FILTERED DATA
# ============================================================

with st.expander(
    "Preview Filtered Corpus",
    expanded=True
):

    display_columns = [
        c for c in df.columns
        if c in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[
            display_columns
        ].head(200),
        use_container_width=True
    )


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

if selected_filter_columns:

    st.subheader(
        "5. Category Distribution"
    )

    distribution_column = st.selectbox(
        "View distribution for",
        options=selected_filter_columns
    )


    distribution = (
        filtered_df[
            distribution_column
        ]
        .value_counts()
        .reset_index()
    )


    distribution.columns = [
        distribution_column,
        "Count"
    ]


    col1, col2 = st.columns(
        [1, 2]
    )


    with col1:

        st.dataframe(
            distribution,
            use_container_width=True,
            hide_index=True
        )


    with col2:

        st.bar_chart(
            distribution.set_index(
                distribution_column
            )
        )


# ============================================================
# SAVE FOR TOPIC MODELLING
# ============================================================

st.divider()

st.subheader(
    "Corpus Ready for Topic Modelling"
)


st.success(
    f"{len(filtered_df):,} documents are ready."
)


# Store for later BERTopic page

st.session_state.filtered_df = filtered_df

st.session_state.text_column = text_column


# ============================================================
# DOWNLOAD FILTERED DATA
# ============================================================

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    "Download Filtered Corpus",
    data=csv_data,
    file_name="filtered_topic_corpus.csv",
    mime="text/csv"
)

# ============================================================
# TOPIC MODELLING
# ============================================================

st.divider()
st.header("6. Topic Modelling")


# ------------------------------------------------------------
# ANALYSIS MODE
# ------------------------------------------------------------

analysis_mode = st.radio(
    "Analysis Mode",
    [
        "Current Filter",
        "Overall Dataset",
        "Compare Categories"
    ],
    horizontal=True
)


# ============================================================
# COMPARISON SETTINGS
# ============================================================

comparison_column_1 = None
comparison_column_2 = None

comparison_values_1 = []
comparison_values_2 = []


if analysis_mode == "Compare Categories":

    st.subheader("Category Comparison")

    st.info(
        "A single shared BERTopic model will be created. "
        "The resulting topics will then be compared across "
        "the selected categories."
    )

    available_comparison_columns = [
        c for c in categorical_columns
        if c != "_topic_text"
    ]

    if len(available_comparison_columns) < 2:

        st.warning(
            "At least two categorical columns are required "
            "for two-dimensional comparison."
        )

    else:

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # PRIMARY CATEGORY
        # ----------------------------------------------------

        with col1:

            default_1 = 0

            # Automatically suggest relevance
            for i, col in enumerate(
                available_comparison_columns
            ):
                if "relevance" in col.lower():
                    default_1 = i
                    break

            comparison_column_1 = st.selectbox(
                "Comparison Dimension 1",
                options=available_comparison_columns,
                index=default_1,
                key="comparison_column_1"
            )


        # ----------------------------------------------------
        # SECONDARY CATEGORY
        # ----------------------------------------------------

        with col2:

            second_options = [
                c
                for c in available_comparison_columns
                if c != comparison_column_1
            ]

            default_2 = 0

            # Automatically suggest sentiment
            for i, col in enumerate(second_options):

                if "sentiment" in col.lower():

                    default_2 = i
                    break

            comparison_column_2 = st.selectbox(
                "Comparison Dimension 2",
                options=second_options,
                index=default_2,
                key="comparison_column_2"
            )


        # ====================================================
        # SELECT CATEGORY VALUES
        # ====================================================

        st.markdown("#### Select Category Values")


        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # VALUES DIMENSION 1
        # ----------------------------------------------------

        with col1:

            values_1 = get_filter_values(
                working_df,
                comparison_column_1
            )

            comparison_values_1 = st.multiselect(
                comparison_column_1,
                options=values_1,
                default=values_1,
                key="comparison_values_1"
            )


        # ----------------------------------------------------
        # VALUES DIMENSION 2
        # ----------------------------------------------------

        with col2:

            values_2 = get_filter_values(
                working_df,
                comparison_column_2
            )

            comparison_values_2 = st.multiselect(
                comparison_column_2,
                options=values_2,
                default=values_2,
                key="comparison_values_2"
            )


        # ====================================================
        # PREVIEW COMBINATIONS
        # ====================================================

        comparison_preview = working_df[
            working_df[comparison_column_1].isin(
                comparison_values_1
            )
            &
            working_df[comparison_column_2].isin(
                comparison_values_2
            )
        ].copy()


        st.markdown(
            "#### Category Combination Preview"
        )


        combination_table = (
            comparison_preview
            .groupby(
                [
                    comparison_column_1,
                    comparison_column_2
                ],
                dropna=False
            )
            .size()
            .reset_index(
                name="Documents"
            )
        )


        st.dataframe(
            combination_table,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL SETTINGS
# ============================================================

with st.expander(
    "BERTopic Settings",
    expanded=False
):

    col1, col2, col3 = st.columns(3)

    with col1:

        embedding_model_name = st.selectbox(
            "Embedding Model",
            [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-multilingual-MiniLM-L12-v2"
            ],
            index=0
        )

    with col2:

        min_topic_size = st.number_input(
            "Minimum Topic Size",
            min_value=5,
            max_value=500,
            value=15,
            step=5
        )

    with col3:

        top_n_words = st.number_input(
            "Keywords per Topic",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )


    col1, col2 = st.columns(2)

    with col1:

        ngram_min = st.selectbox(
            "Minimum N-gram",
            [1, 2],
            index=0
        )

    with col2:

        ngram_max = st.selectbox(
            "Maximum N-gram",
            [1, 2, 3],
            index=1
        )


# ============================================================
# SELECT CORPUS
# ============================================================

if analysis_mode == "Current Filter":

    modelling_df = filtered_df.copy()

    st.info(
        f"BERTopic will analyse the current filtered corpus "
        f"containing {len(modelling_df):,} documents."
    )


elif analysis_mode == "Overall Dataset":

    modelling_df = working_df.copy()

    st.info(
        f"BERTopic will analyse all "
        f"{len(modelling_df):,} valid documents."
    )


else:

    if (
        comparison_column_1
        and comparison_column_2
        and comparison_values_1
        and comparison_values_2
    ):

        modelling_df = working_df[
            working_df[
                comparison_column_1
            ].isin(comparison_values_1)
            &
            working_df[
                comparison_column_2
            ].isin(comparison_values_2)
        ].copy()

    else:

        modelling_df = working_df.iloc[0:0].copy()


    st.info(
        f"Shared BERTopic model will analyse "
        f"{len(modelling_df):,} documents."
    )


# ------------------------------------------------------------
# LARGE DATA WARNING
# ------------------------------------------------------------

if len(modelling_df) > 20000:

    st.warning(
        f"You selected {len(modelling_df):,} documents. "
        "BERTopic may take considerable time on CPU."
    )


# ------------------------------------------------------------
# RUN BUTTON
# ------------------------------------------------------------

run_model = st.button(
    "Run Topic Modelling",
    type="primary",
    use_container_width=True
)


if run_model:

    if len(modelling_df) < 20:

        st.error(
            "Too few documents for reliable topic modelling."
        )

        st.stop()


    with st.status(
        "Running BERTopic...",
        expanded=True
    ) as status:

        try:

            st.write(
                f"Preparing {len(modelling_df):,} documents..."
            )

            st.write(
                f"Embedding model: {embedding_model_name}"
            )

            st.write(
                "Generating document embeddings..."
            )

            result = run_topic_model(
                modelling_df,
                text_column="_topic_text",
                embedding_model_name=embedding_model_name,
                min_topic_size=min_topic_size,
                ngram_min=ngram_min,
                ngram_max=ngram_max,
                top_n_words=top_n_words
            )

            st.session_state.topic_result = result

            st.session_state.topic_analysis_mode = (
                analysis_mode
            )

            status.update(
                label="Topic modelling completed.",
                state="complete"
            )

        except Exception as e:

            status.update(
                label="Topic modelling failed.",
                state="error"
            )

            st.exception(e)


# ============================================================
# TWO-DIMENSIONAL CATEGORY COMPARISON
# ============================================================

if (
    "topic_result" in st.session_state
    and st.session_state.get(
        "topic_analysis_mode"
    ) == "Compare Categories"
):

    st.divider()

    st.header(
        "8. Topic × Category Analysis"
    )


    result = st.session_state.topic_result

    comparison_df = result["data"].copy()


    # ========================================================
    # VALIDATION
    # ========================================================

    if (
        comparison_column_1 is None
        or comparison_column_2 is None
    ):

        st.warning(
            "Select two comparison dimensions."
        )

    else:

        # ====================================================
        # REMOVE OUTLIERS OPTION
        # ====================================================

        exclude_outliers = st.checkbox(
            "Exclude BERTopic outliers (Topic -1)",
            value=True
        )


        analysis_df = comparison_df.copy()


        if exclude_outliers:

            analysis_df = analysis_df[
                analysis_df["topic_id"] != -1
            ]


        # ====================================================
        # CREATE COMBINATION LABEL
        # ====================================================

        analysis_df[
            "_category_combination"
        ] = (
            analysis_df[
                comparison_column_1
            ].astype(str)
            +
            " | "
            +
            analysis_df[
                comparison_column_2
            ].astype(str)
        )


        # ====================================================
        # TOPIC × CATEGORY COUNTS
        # ====================================================

        topic_category = (
            analysis_df
            .groupby(
                [
                    "topic_id",
                    comparison_column_1,
                    comparison_column_2
                ],
                dropna=False
            )
            .size()
            .reset_index(
                name="Documents"
            )
        )


        st.subheader(
            "Topic Distribution by Category"
        )


        st.dataframe(
            topic_category,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # PERCENTAGE WITHIN CATEGORY
        # ====================================================

        category_totals = (
            topic_category
            .groupby(
                [
                    comparison_column_1,
                    comparison_column_2
                ]
            )["Documents"]
            .transform("sum")
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
        )


        topic_category[
            "Percentage"
        ] = (
            topic_category[
                "Percentage"
            ].round(2)
        )


        st.subheader(
            "Topic Prevalence Within Each Category"
        )


        st.dataframe(
            topic_category,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # SELECT TOPIC
        # ====================================================

        available_comparison_topics = sorted(
            analysis_df[
                "topic_id"
            ].unique()
        )


        selected_comparison_topic = (
            st.selectbox(
                "Select Topic to Compare",
                options=available_comparison_topics,
                key="comparison_topic_selector"
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
        # ====================================================
        # HEATMAP
        # ====================================================

        st.markdown(
            "#### Category Heatmap"
        )


        fig = px.imshow(
            topic_matrix,
            text_auto=True,
            aspect="auto",
            labels={
                "x": comparison_column_2,
                "y": comparison_column_1,
                "color": "Documents"
            },
            title=(
                f"Topic {selected_comparison_topic}: "
                f"{comparison_column_1} × "
                f"{comparison_column_2}"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ====================================================
        # MATRIX
        # ====================================================

        topic_matrix = (
            selected_topic_df
            .pivot_table(
                index=comparison_column_1,
                columns=comparison_column_2,
                values="Documents",
                fill_value=0
            )
        )


        st.markdown(
            f"### Topic {selected_comparison_topic}"
        )


        st.markdown(
            "#### Document Counts"
        )


        st.dataframe(
            topic_matrix,
            use_container_width=True
        )


        # ====================================================
        # BAR CHART
        # ====================================================

        st.markdown(
            "#### Category Distribution"
        )


        chart_df = (
            selected_topic_df.copy()
        )


        chart_df[
            "Category"
        ] = (
            chart_df[
                comparison_column_1
            ].astype(str)
            +
            " | "
            +
            chart_df[
                comparison_column_2
            ].astype(str)
        )


        chart_df = (
            chart_df[
                [
                    "Category",
                    "Documents"
                ]
            ]
            .set_index(
                "Category"
            )
        )


        st.bar_chart(
            chart_df
        )

        # ----------------------------------------------------
        # REPRESENTATIVE DOCUMENTS
        # ----------------------------------------------------

        st.markdown(
            "### Representative Documents"
        )


        representative_docs = (
            get_representative_documents(
                topic_model,
                selected_topic
            )
        )


        if representative_docs:

            for i, document in enumerate(
                representative_docs,
                start=1
            ):

                with st.expander(
                    f"Representative Document {i}"
                ):

                    st.write(document)

        else:

            st.info(
                "No representative documents available."
            )


    # --------------------------------------------------------
    # DOWNLOAD RESULTS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Export Topic Results"
    )


    topic_csv = result_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "Download Dataset with Topic IDs",
        data=topic_csv,
        file_name="topic_modelling_results.csv",
        mime="text/csv"
    )
