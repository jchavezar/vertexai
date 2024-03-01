#%%
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel

project_id = "vtxdemos"
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

# from google.cloud import bigquery
client = bigquery.Client(project=project_id)

sql = """
SELECT text
FROM `bigquery-public-data.hacker_news.full`
LIMIT 10
"""

df = client.query(sql).to_dataframe()

df["embedding_v1"] = df["text"].apply(lambda x: model.get_embeddings([x])[0].values)
x=bigquery.Dataset("vtxdemos").table("text_embeddings")
bigquery.Client(project_id=project_id).create_table()