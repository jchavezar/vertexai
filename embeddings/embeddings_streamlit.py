#%%
# Libraries
from google.cloud import bigquery
# Variables
project_id: str = "vtxdemos"

# Define
client = bigquery.Client(project_id)

#text = st.text_input("Enter your question:")

# Create the table
query = """
CREATE OR REPLACE TABLE vector_search.stackoverflow_embeddings AS
SELECT
content,
ml_generate_embedding_result
FROM
ML.GENERATE_EMBEDDING(
  MODEL `vector_search.embeddings_model`,
  (SELECT title AS content from `bigquery-public-data.stackoverflow.posts_questions` LIMIT 10000))
"""

job = client.query(query)
job.result()
