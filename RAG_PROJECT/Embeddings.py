import requests

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "bge-m3"


def create_embeddings(text_list, batch_size=32):
   
    all_embeddings = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        batch = [t[:4000] for t in batch]  # keep your existing truncation safeguard

        print(f"Embedding batch {i} to {i + len(batch)}")

        r = requests.post(OLLAMA_URL, json={
            "model": EMBED_MODEL,
            "input": batch
        }, timeout=600)

        data = r.json()

        if "error" in data:
            raise Exception(data["error"])
        if "embeddings" not in data:
            raise Exception(data)

        all_embeddings.extend(data["embeddings"])

    return all_embeddings