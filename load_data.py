from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pandas as pd
from tqdm.notebook import tqdm
# Import client library
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
if __name__ == "__main__":

    model = SentenceTransformer(
        "all-MiniLM-L6-v2", device="cpu"
    )  # or device="cpu" if you don't have a GPU

    df = pd.read_json("./startups_demo.json", lines=True)
    vectors = model.encode(
        [row.alt + ". " + row.description for row in df.itertuples()],
        show_progress_bar=True,
    )
    vectors.shape
    # > (40474, 384)
    np.save("startup_vectors.npy", vectors, allow_pickle=False)

    qdrant_client = QdrantClient("http://localhost:6333")
    qdrant_client.recreate_collection(
        collection_name="startups",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    fd = open("./startups_demo.json")

    # payload is now an iterator over startup data
    payload = map(json.loads, fd)

    # Load all vectors into memory, numpy array works as iterable for itself.
    # Other option would be to use Mmap, if you don't want to load all data into RAM
    vectors = np.load("./startup_vectors.npy")
    qdrant_client.upload_collection(
        collection_name="startups",
        vectors=vectors,
        payload=payload,
        ids=None,  # Vector ids will be assigned automatically
        batch_size=256,  # How many vectors will be uploaded in a single request?
    )