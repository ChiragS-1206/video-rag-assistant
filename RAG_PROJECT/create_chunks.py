import whisper
import json 
import os 


model = whisper.load_model("large-v2")

audios =  os.listdir("AUDIOS")


for audio in audios:
    number = audio.split("_")[0]
    title = audio.split("_")[1][:-4]
    print(f"{number} is title:- {title}")

    result = model.transcribe(audio= f"AUDIOS/{audio}",language= "hi",task ="translate",fp16=False)

    chunks =[]

    for segement in result["segments"]:
        chunks.append({"number":number,"title": title,"start": segement["start"] , "end": segement["end"],"text": segement["text"]})


    chunk_with_metadeta = {"chunks" : chunks ,"text": result["text"]}


    with open(f"jsons/{audio}.json","w") as f:
        json.dump(chunk_with_metadeta,f)
