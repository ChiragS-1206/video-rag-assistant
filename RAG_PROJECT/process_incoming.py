import os
import re
import requests
import ast
import faiss

import pandas as pd
import numpy as np

from google import genai
from dotenv import load_dotenv
from Embeddings import create_embeddings
# from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
        raise EnvironmentError("gemini key not found")




client = genai.Client(api_key=GEMINI_API_KEY)


INDEX_FILE = "course.index"
CSV_FILE = "output2.csv"
TOP_K = 3


def inference_gemini(prompt):
        response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
        )

        return {"response":response.text}





def sec_to_min(seconds):
        seconds = int(seconds)
        min = seconds//60
        sec= seconds%60
        return f"{min}:{sec:02d}"


def split_query(query):
        parts = re.split(r"\band\b|,|\bor\b",query,flags=re.IGNORECASE)
        # return [p.split() for p in parts if p.split()]
        return [p.strip() for p in parts if p.strip()]


def load_data_index():  
        df = pd.read_csv(CSV_FILE)
        # csv file embedding ko string mei convert krdeti h but hume toh list chiaye for process so we do this
        df["embedding"] = df["embedding"].apply(ast.literal_eval)
        index = faiss.read_index(INDEX_FILE)

        return df,index
def search(query_text,df,index,top_k):
        q_embedding = np.array(create_embeddings([query_text])).astype("float32")
        faiss.normalize_L2(q_embedding)

        distances, indices = index.search(q_embedding,top_k)
        return indices[0]


def main():
        df,index= load_data_index()
        incoming_query = input(" enter ur question:- ")
        queries = split_query(incoming_query)
        all_query = []

        for q in queries:
                macthed_queries = search(q,df,index,TOP_K)
                all_query.extend(macthed_queries)

        # to remove duplicates
        all_query = list(dict.fromkeys(all_query))
        new_df = df.loc[all_query].copy()
        new_df["start"] = new_df["start"].apply(sec_to_min)
        new_df["end"] = new_df["end"].apply(sec_to_min)

        chunks_json = new_df[["title", "number", "text", "start", "end"]].to_json(
        orient="records",
        indent=2
        )

        prompt = f"""
        You are a Python course assistant.

        IMPORTANT RULES:
        - Do NOT write code.
        - Do NOT explain implementation.
        - The user may ask multiple topics in one question.
        - Answer each topic separately.
        - Use only the given course chunks.
        - If one topic is found but another topic is missing, answer the found topic and say the missing topic was not found in the provided chunks.
        - Give video number, title, and timestamp for each topic.

        User question:
        {incoming_query}

        Relevant course chunks:
        {chunks_json}

        Answer format:

        Topic 1:
        Answer:
        Where taught:
        What to watch:

        Topic 2:
        Answer:
        Where taught:
        What to watch:
        """

        response = inference_gemini(prompt)["response"]
        print(response)


        

        with open("promt.txt","w") as f:
                f.write(prompt)

        with open("reponse.txt","w") as f:
                f.write(response)


if __name__ == "__main__":
        main()















