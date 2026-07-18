import os
import json
import math

n =5
for filename in os.listdir("jsons"):
    if filename.endswith(".json"):
        file_path = os.path.join("jsons",filename)
        with open(file_path,"r",encoding="utf-8") as f:
            data = json.load(f)
            new_chunks = []
            num_chunks = len(data["chunks"])
            num_grp = math.ceil(num_chunks/n);

            for i in range(num_grp):
                start_idx = i*n
                end_idx = min((i+1)*n,num_chunks)

                chunks_grp = data["chunks"][start_idx:end_idx]


                new_chunks.append({
                    "number":data["chunks"][0]["number"],
                    "title" :chunks_grp[0]["title"],
                    "start": chunks_grp[0]["start"],
                    "end" : chunks_grp[-1]["end"],
                    "text": " ".join(c["text"] for c in chunks_grp)

                })

            
            os.makedirs("new_json",exist_ok=True)
            with open(os.path.join("new_json",filename),"w",encoding="utf-8") as f:
                json.dump({"chunks": new_chunks, "text":data["text"]},f,indent=4)

