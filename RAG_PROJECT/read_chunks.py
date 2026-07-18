import json
import os

import faiss
import numpy as np
import pandas as pd

from Embeddings import create_embeddings



INDEX_FILE = "course.index"
CSV_FILE = "output2.csv"


def build_index():
    new_json_dir = "new_json"
    new_json_files = os.listdir(new_json_dir)

    my_dicts = []
    chunk_id = 0

    for json_file in new_json_files:
        with open(os.path.join(new_json_dir, json_file), "r") as f:
            content = json.load(f)

        texts = [c["text"] for c in content["chunks"]]
        embeddings = create_embeddings(texts)  

        for i, chunk in enumerate(content["chunks"]):
            chunk["chunk_id"] = chunk_id
            chunk_id += 1
            chunk["embedding"] = embeddings[i]
            my_dicts.append(chunk)

    print(f"Total chunks embedded: {len(my_dicts)}")

    data = pd.DataFrame.from_records(my_dicts)
    data.to_csv(CSV_FILE, index=False)
    print("Saved csv ")

    # ---- Build FAISS index ----

    emb_matrix = np.array(data["embedding"].tolist()).astype("float32")


    faiss.normalize_L2(emb_matrix)

    index = faiss.IndexFlatIP(emb_matrix.shape[1])
    index.add(emb_matrix)
    faiss.write_index(index, INDEX_FILE)

    print("Saved FAISS index ")


if __name__ == "__main__":
    build_index()