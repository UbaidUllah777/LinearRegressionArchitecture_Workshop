
"""
Preprocessing Module
Linear Regression Architecture Workshop

Provides reusable functions for:
1. Data cleaning
2. Feature and target selection
3. Train/test splitting
4. Feature standardization
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Project Root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# 1. Data Cleaning
# --------------------------------------------------

def clean_data(df, feature="MedInc", target="MedHouseVal"):
    """
    Select the predictor and target variables.

    Convert values to numeric and remove observations
    containing missing or invalid values.
    """

    # Check whether required columns exist
    required_columns = [feature, target]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select only the required columns
    cleaned_df = df[required_columns].copy()

    # Convert values to numeric
    for column in required_columns:
        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column],
            errors="coerce"
        )

    # Replace infinite values with missing values
    cleaned_df = cleaned_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove missing predictor or target values
    cleaned_df = cleaned_df.dropna(
        subset=required_columns
    )

    return cleaned_df


# --------------------------------------------------
# 2. Train/Test Split and Standardization
# --------------------------------------------------

def prepare_data(
    df,
    feature="MedInc",
    target="MedHouseVal",
    test_size=0.20,
    random_state=42
):
    """
    Clean the dataset, split it into training/testing
    subsets, and standardize the predictor.

    The scaler is fitted using training data only
    to prevent data leakage.
    """

    # Step 1: Clean the data
    cleaned_df = clean_data(
        df,
        feature=feature,
        target=target
    )

    if cleaned_df.empty:
        raise ValueError(
            "No valid observations remain after cleaning."
        )

    # Step 2: Select X and y
    X = cleaned_df[[feature]]
    y = cleaned_df[target]

    # Step 3: Split into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Step 4: Create the scaler
    scaler = StandardScaler()

    # Fit using training data only
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same transformation to testing data
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Return all prepared data
    return {
        "cleaned_data": cleaned_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler
    }


# --------------------------------------------------
# Direct Execution
# --------------------------------------------------

if __name__ == "__main__":

    print("Testing Preprocessing Module...")

    # Load our previously collected California CSV
    csv_path = (
        PROJECT_ROOT / "data" / "raw" /
        "california_housing.csv"
    )

    df = pd.read_csv(csv_path)

    # Run the complete preprocessing function
    prepared = prepare_data(df)

    print("\nPreprocessing completed successfully!")

    print(
        "Cleaned dataset shape:",
        prepared["cleaned_data"].shape
    )

    print(
        "Training shape:",
        prepared["X_train"].shape
    )

    print(
        "Testing shape:",
        prepared["X_test"].shape
    )

    print(
        "Scaled training mean:",
        round(prepared["X_train_scaled"].mean(), 4)
    )

    print(
        "Scaled training standard deviation:",
        round(prepared["X_train_scaled"].std(), 4)
    )
