# app.py
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# Load the CSV files and model ONCE
file_paths = [
    "Server_Method/deployments.csv",
    "Server_Method/device_info.csv",
    "Server_Method/property.csv",
    "Server_Method/device_category.csv",
    "Server_Method/locations.csv",
    "Server_Method/deployments_2.csv",
    "Server_Method/device_info_2.csv",
    "Server_Method/property_2.csv",
    "Server_Method/device_category_2.csv",
    "Server_Method/locations_2.csv",
    "Server_Method/data_and_knowledge_queries.csv"
]

dfs = [pd.read_csv(path) for path in file_paths]
df_all = pd.concat(dfs, ignore_index=True)

questions = df_all["question"].tolist()
labels = df_all["label"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(questions, normalize_embeddings=True)
embeddings_tensor = torch.tensor(embeddings)

# Set up FastAPI
app = FastAPI()

# Input schema
class QueryInput(BaseModel):
    query: str
    top_k: int = 1

# Output schema
class QueryOutput(BaseModel):
    label: str

# Endpoint
@app.post("/classify", response_model=QueryOutput)
def classify_query(data: QueryInput):
    query_emb = model.encode(data.query, normalize_embeddings=True)
    similarities = util.cos_sim(torch.tensor(query_emb), embeddings_tensor)[0]
    top_index = similarities.topk(data.top_k).indices[0].item()
    label = labels[top_index]
    return QueryOutput(label=label)

@app.post("/test", response_model=QueryOutput)
def test(data: QueryInput):
    label = "test"
    return QueryOutput(label=label)
