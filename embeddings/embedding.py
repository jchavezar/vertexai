#%%
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel

project_id = "vtxdemos"
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

# EDA
client = bigquery.Client(project=project_id)
sql = """
SELECT text
FROM `hacker_news.full`
LIMIT 10
"""

df = client.query(sql).to_dataframe()

# CREATE MODEL FROM CONNECTION
sql = """CREATE OR REPLACE MODEL vector_search.embeddings_model
  REMOTE WITH CONNECTION `us.embeddings`
  OPTIONS (remote_service_type = 'CLOUD_AI_TEXT_EMBEDDING_MODEL_V1');"""
job = client.query(sql)
job.result()

#%%
# CREATE NEW TABLE WITH EMBEDDINGS
sql = """
CREATE OR REPLACE TABLE vector_search.hacker_news_embeddings AS
SELECT
content,
ml_generate_embedding_result
FROM
ML.GENERATE_EMBEDDING(
  MODEL `vector_search.embeddings_model`,
  (SELECT text AS content from `bigquery-public-data.hacker_news.full` LIMIT 10))"""

job = client.query(sql)
job.result()

#%%
# SEARCH SIMILARITY
sql = """
WITH hacker_news AS (
    SELECT * FROM
        ML.GENERATE_EMBEDDING(
            MODEL `vector_search.embeddings_model`,
            (SELECT text as content from `bigquery-public-data.hacker_news.full` LIMIT 10),
            STRUCT(TRUE AS flatten_json_output)
        )
),
    text_query AS (
        SELECT ml_generate_embedding_result
        FROM
            ML.GENERATE_EMBEDDING(
                MODEL `vector_search.embeddings_model`,
                (SELECT "What exactly are you looking for? I think Pytorch is so ergonomic that there is no need for other resources. The only gotchas I found were in the data loading utilities." AS content),
                STRUCT(TRUE AS flatten_json_output)
                )
    )
SELECT
    content,
    ML.DISTANCE(
        (SELECT ml_generate_embedding_result FROM text_query),
        ml_generate_embedding_result,
        "COSINE"
        ) AS distance_to_average_review
FROM hacker_news
ORDER BY distance_to_average_review
"""
job = client.query(sql)
result = job.result()

response = result.to_dataframe()