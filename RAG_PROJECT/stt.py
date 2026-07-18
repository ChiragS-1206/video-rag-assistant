import whisper

model= whisper.load_model("large-v2")\

result = model.transcribe(audio= "AUDIOS/typecasting_10sec.mp3",language= "hi",task ="translate",fp16=False)

print(result)


chunks =[]

for segement in result["segments"]:
    chunks.append({"start": segement["start"] , "end": segement["end"],"text": segement["text"]})

print(chunks)


