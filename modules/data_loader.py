import pandas as pd
from pathlib import Path


def load_csv(uploaded_file):
    """
    Load an uploaded CSV file.
    Attempts several common encodings.
    """

    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    last_error = None

    for encoding in encodings:
        try:
            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                encoding=encoding,
                low_memory=False
            )

            return df, encoding

        except Exception as e:
            last_error = e

    raise ValueError(
        f"Unable to read CSV file. Last error: {last_error}"
    )


def get_column_summary(df):
    """
    Produce basic information about each column.
    """

    summary = []

    total_rows = len(df)

    for column in df.columns:

        unique_count = df[column].nunique(dropna=True)
        missing_count = df[column].isna().sum()

        if total_rows > 0:
            missing_percentage = (
                missing_count / total_rows
            ) * 100
        else:
            missing_percentage = 0

        summary.append({
            "Column": column,
            "Data Type": str(df[column].dtype),
            "Unique Values": unique_count,
            "Missing": missing_count,
            "Missing %": round(missing_percentage, 2)
        })

    return pd.DataFrame(summary)


def detect_text_columns(df):
    """
    Guess which columns are likely to contain text.

    A text column normally:
    - has object/string datatype
    - contains reasonably long text
    """

    candidates = []

    for column in df.columns:

        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):

            sample = (
                df[column]
                .dropna()
                .astype(str)
                .head(500)
            )

            if len(sample) == 0:
                continue

            average_length = sample.str.len().mean()

            if average_length >= 20:
                candidates.append(column)

    return candidates


def detect_categorical_columns(
    df,
    max_unique=100
):
    """
    Identify columns suitable for dynamic filtering.

    Examples:
    sentiment
    emotion
    source
    subreddit
    gender
    label
    year
    """

    categorical_columns = []

    for column in df.columns:

        unique_count = df[column].nunique(dropna=True)

        if 1 < unique_count <= max_unique:
            categorical_columns.append(column)

    return categorical_columns