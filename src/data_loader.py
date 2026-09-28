
"""
Data Loading Module
Linear Regression Architecture Workshop

Provides reusable functions for loading data from:
1. Local CSV files
2. Statistics Canada API
3. PostgreSQL database
"""

import yaml

import os
import io
import zipfile

from pathlib import Path

import pandas as pd
import requests

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# --------------------------------------------------
# Project paths
# --------------------------------------------------

# Identify the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# Configuration Loader
# --------------------------------------------------

def load_config(
    config_path="configs/experiment_config.yaml"
):
    """
    Load experiment settings from a YAML configuration file.

    Relative paths are resolved from the project root.
    """

    path = Path(config_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The configuration must contain a YAML mapping."
        )

    return config



# --------------------------------------------------
# 1. CSV Loader
# --------------------------------------------------

def load_csv(file_path):
    """
    Load a CSV file into a Pandas DataFrame.

    Relative paths are resolved from the project root.
    """

    path = Path(file_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    return pd.read_csv(path)


# --------------------------------------------------
# 2. API Loader
# --------------------------------------------------

def load_statcan_api(
    api_url,
    csv_filename="18100205.csv",
    archive_path=None
):
    """
    Retrieve a Statistics Canada table through its API.

    The API returns a ZIP download URL. The ZIP is
    downloaded and its main CSV file is loaded into
    a Pandas DataFrame.

    Optionally save the original ZIP archive.
    """

    # Request the downloadable table URL
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()

    api_result = response.json()

    download_url = api_result["object"]

    # Download the ZIP archive
    data_response = requests.get(
        download_url,
        timeout=120
    )

    data_response.raise_for_status()

    # Optionally save the original ZIP file
    if archive_path is not None:

        archive_path = Path(archive_path)

        if not archive_path.is_absolute():
            archive_path = PROJECT_ROOT / archive_path

        archive_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        archive_path.write_bytes(data_response.content)

    # Read the requested CSV from the ZIP archive
    with zipfile.ZipFile(
        io.BytesIO(data_response.content)
    ) as zip_file:

        with zip_file.open(csv_filename) as csv_file:
            df = pd.read_csv(
                csv_file,
                low_memory=False
            )

    return df


# --------------------------------------------------
# 3. PostgreSQL Loader
# --------------------------------------------------

def load_postgres(query, params=None):
    """
    Execute a SQL SELECT query against PostgreSQL
    and return the result as a Pandas DataFrame.

    Database credentials are loaded securely from
    the project's .env file.
    """

    # Load environment variables
    load_dotenv(PROJECT_ROOT / ".env")

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL was not found in the .env file."
        )

    # Create a database engine
    engine = create_engine(database_url)

    try:
        with engine.connect() as connection:

            df = pd.read_sql_query(
                text(query),
                connection,
                params=params
            )

        return df

    finally:
        engine.dispose()


# --------------------------------------------------
# Direct Execution
# --------------------------------------------------

if __name__ == "__main__":

    print("Testing Data Loading Module...")

    # Test local CSV loading without requiring
    # an internet or database connection.
    california_df = load_csv(
        "data/raw/california_housing.csv"
    )

    print("\nCalifornia CSV loaded successfully!")

    print("Dataset shape:", california_df.shape)

    print("\nFirst five records:")
    print(california_df.head())
