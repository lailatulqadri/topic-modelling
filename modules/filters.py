import pandas as pd


def apply_category_filters(
    df,
    selected_filters
):
    """
    Apply multiple category filters.

    selected_filters example:

    {
        "sentiment": ["Positive", "Negative"],
        "source": ["Reddit"],
        "year": [2025, 2026]
    }
    """

    filtered_df = df.copy()

    for column, selected_values in selected_filters.items():

        if not selected_values:
            continue

        filtered_df = filtered_df[
            filtered_df[column].isin(selected_values)
        ]

    return filtered_df


def prepare_text_data(
    df,
    text_column
):
    """
    Remove empty text records and create
    a clean internal text field.
    """

    working_df = df.copy()

    working_df = working_df[
        working_df[text_column].notna()
    ].copy()

    working_df["_topic_text"] = (
        working_df[text_column]
        .astype(str)
        .str.strip()
    )

    working_df = working_df[
        working_df["_topic_text"].str.len() > 0
    ]

    return working_df


def get_filter_values(
    df,
    column
):
    """
    Return unique values safely for Streamlit.
    """

    values = (
        df[column]
        .dropna()
        .unique()
        .tolist()
    )

    try:
        values = sorted(values)
    except TypeError:
        values = sorted(
            values,
            key=lambda x: str(x)
        )

    return values