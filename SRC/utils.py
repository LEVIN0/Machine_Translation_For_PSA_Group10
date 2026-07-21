import pandas as pd


def save_dataframe(df, filepath):
    """
    Save a DataFrame as CSV.
    """
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")


def preview(df, rows=5):
    """
    Display the first few rows.
    """
    return df.head(rows)