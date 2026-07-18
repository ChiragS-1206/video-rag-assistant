import os
import subprocess


files = os.listdir("VIDEOS")
for file in files:
    day_no = file.split(" #")[1].split(".")[0]
    file_name = file.split("_ Python")[0]
    subprocess.run(["ffmpeg","-i",f"VIDEOS/{file}",f"AUDIOS/{day_no}_{file_name}.mp3"])

    