"""
BigQueryManager Module

This module provides a class to manage interactions with Google BigQuery, allowing execution 
of SQL queries and optional storage of results in a specified destination table.

Classes:
    - BigQueryManager: Handles query execution and result retrieval.

Usage Example:
    bq_manager = BigQueryManager(project_id, dataset_id)
    query = "SELECT * FROM dataset.table LIMIT 10"
    result_df = bq_manager.execute_query(query)

Environment Variables:
    - GOOGLE_APPLICATION_CREDENTIALS: Path to the GCP service account key file.
    - PROJECT_ID: GCP project ID.
    - DATASET_ID: BigQuery dataset ID.
"""

import os
from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud.bigquery.table import RowIterator

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GCP_SERVICE_ACCOUNT_JSON_KEY_PATH"
)

project_id = os.getenv("PROJECT_ID")
dataset_id = os.getenv("DATASET_ID")


class BigQueryManager:
    """
    A manager class to interact with Google BigQuery, providing functionality to execute SQL queries
    and optionally save query results to a specified table.

    Attributes:
        project_id (str): The GCP project ID.
        dataset_id (str): The BigQuery dataset ID.
        client (bigquery.Client): An instance of the BigQuery client.

    Methods:
        execute_query(query: str, destination_table: str = None):
            Executes the provided SQL query and returns results
            as a DataFrame if no destination table is specified.
    """

    def __init__(self, project_id, dataset_id):
        """
        Initializes the BigQueryManager with project and dataset IDs.

        Args:
            project_id (str): The GCP project ID.
            dataset_id (str): The BigQuery dataset ID.
        """

        self.client = bigquery.Client()
        self.project_id = project_id
        self.dataset_id = dataset_id

    def execute_query(self, query, destination_table=None):
        """
        Run a query. Optionally save the results to a table or return the result as a DataFrame.
        """
        job_config = bigquery.QueryJobConfig()

        # Handle destination table for non-DDL queries
        if destination_table and not query.strip().lower().startswith(
            ("create", "alter")
        ):
            table_ref = f"{self.project_id}.{self.dataset_id}.{destination_table}"
            job_config.destination = table_ref
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

        query_job = self.client.query(query, job_config=job_config)
        result: RowIterator = query_job.result()  # Wait for the query to complete

        # Return DataFrame if no destination_table is provided
        if not destination_table:
            return result.to_dataframe()


# Usage
if __name__ == "__main__":

    # Instantiate BigQueryManager with the project and dataset IDs
    bq_manager = BigQueryManager(project_id=project_id, dataset_id=dataset_id)

    # Example: Run a query to create or fetch data
    QUERY = """
    SELECT d.County, AVG(d.Income) AS AverageIncome
    FROM llm-testing-447813.us_census.demographics AS d
    ORDER BY d.SelfEmployed DESC
    LIMIT 5;"""
    data = bq_manager.execute_query(QUERY)
    print(data)
    print(type(data))
    print(data.info())
