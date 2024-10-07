# utils.py

import pandas as pd
from langchain.docstore.document import Document
import streamlit as st
import os
import logging

logger = logging.getLogger(__name__)

@st.cache_data
def load_data(csv_file):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        if not os.path.exists(csv_file):
            st.error("CSV file does not exist at the specified path.")
            st.stop()
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded data from {csv_file}.")
        return df
    except FileNotFoundError:
        st.error("The CSV file was not found. Please check the CSV_FILE_PATH.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()

def combine_columns(row):
    """
    Combines relevant columns from a DataFrame row into a single string.

    Args:
        row (pd.Series): A row from the DataFrame.

    Returns:
        str: A formatted string containing the strain information.
    """
    return f"""
    Strain: {row['Strain']}
    Type: {row['Type']}
    Rating: {row['Rating']}
    Effects: {row['Effects']}
    Flavor: {row['Flavor']}
    Description: {row['Description']}
    """

def create_documents(df):
    """
    Creates a list of Document objects from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing strain data.

    Returns:
        List[Document]: A list of Document objects.
    """
    documents = []
    for _, row in df.iterrows():
        text = combine_columns(row)
        metadata = {
            'Strain': row.get('Strain', 'N/A'),
            'Type': row.get('Type', 'N/A'),
            'Rating': row.get('Rating', 'N/A'),
            'Effects': row.get('Effects', 'N/A'),
            'Flavor': row.get('Flavor', 'N/A'),
        }
        documents.append(Document(page_content=text, metadata=metadata))
    logger.info(f"Created {len(documents)} Document objects.")
    return documents
