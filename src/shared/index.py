import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


_metadata_df = pd.read_csv("data/raw/openaccess/MetObjects.csv")
_metadata_df = _metadata_df[_metadata_df["Is Public Domain"] == True].reset_index(drop=True)
_metadata_df = _metadata_df.set_index("Object ID", drop=False)

def get_metadata(object_id: int) -> pd.Series:
    return _metadata_df.loc[object_id]

EMBED_FIELDS = [
    "Title", "Object Name", "Culture", "Period", "Dynasty",
    "Artist Display Name", "Artist Nationality", "Object Date",
    "Medium", "Classification", "Department", "Tags", "Country",
]

def build_embedding_text(row: pd.Series) -> str:
    parts = []
    for field in EMBED_FIELDS:
        value = row.get(field)
        if pd.notna(value) and str(value).strip():
            parts.append(f"{field}: {value}")
    return ". ".join(parts)

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = SentenceTransformer("all-mpnet-base-v2", device=_device)

_embeddings = np.load("data/processed/curator_embeddings.npy")
_object_ids = np.load("data/processed/curator_object_ids.npy")

_norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
_normalized_embeddings = _embeddings / _norms

def search(query: str, top_k: int = 25):
    query_vec = _model.encode(query, normalize_embeddings=True)
    scores = _normalized_embeddings @ query_vec
    top_indices = np.argsort(-scores)[:top_k]
    return [(int(_object_ids[i]), float(scores[i])) for i in top_indices]