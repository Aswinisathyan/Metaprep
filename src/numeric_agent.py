import numpy as np
import pandas as pd


# ======================================================
# NUMERIC QUALITY METRIC
# ======================================================

def compute_numeric_quality(df, numeric_cols):

    if len(numeric_cols) == 0:
        return 1.0

    completeness_scores = []
    variance_scores = []

    for col in numeric_cols:

        # Completeness
        completeness = 1 - df[col].isnull().mean()
        completeness_scores.append(completeness)

        # Variance Stability (avoid zero variance)
        var = df[col].var()
        variance_scores.append(min(var, 1))

    C_score = np.mean(completeness_scores)
    V_score = np.mean(variance_scores)

    # Final Numeric Quality
    NQ = 0.6 * C_score + 0.4 * V_score

    return round(NQ, 3)


# ======================================================
# NUMERIC AGENT
# ======================================================

def numeric_agent(df):

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        return df, "No Numeric Columns", 1.0

    best_df = df.copy()
    best_score = -1
    best_pipeline = None

    # ------------------------------
    # PIPELINE A: Mean + MinMax
    # ------------------------------
    df_a = df.copy()

    for col in numeric_cols:
        df_a[col] = df_a[col].fillna(df_a[col].mean())
        minv = df_a[col].min()
        maxv = df_a[col].max()
        if maxv != minv:
            df_a[col] = (df_a[col] - minv) / (maxv - minv)

    score_a = compute_numeric_quality(df_a, numeric_cols)

    if score_a > best_score:
        best_score = score_a
        best_df = df_a.copy()
        best_pipeline = "Mean Imputation + MinMax Scaling"

    # ------------------------------
    # PIPELINE B: Median + Robust
    # ------------------------------
    df_b = df.copy()

    for col in numeric_cols:
        df_b[col] = df_b[col].fillna(df_b[col].median())
        q1 = df_b[col].quantile(0.25)
        q3 = df_b[col].quantile(0.75)
        iqr = q3 - q1
        if iqr != 0:
            df_b[col] = (df_b[col] - df_b[col].median()) / iqr

    score_b = compute_numeric_quality(df_b, numeric_cols)

    if score_b > best_score:
        best_score = score_b
        best_df = df_b.copy()
        best_pipeline = "Median Imputation + Robust Scaling"

    return best_df, best_pipeline, best_score