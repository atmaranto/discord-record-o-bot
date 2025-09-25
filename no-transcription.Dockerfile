FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y ffmpeg curl python3 python3-pip bash-completion git libportaudio2 portaudio19-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

ADD no-transcription-requirements.txt /requirements.txt
RUN python3 -m pip install --break -r /requirements.txt

ADD record.py /record.py

CMD ["python3", "/record.py"]
