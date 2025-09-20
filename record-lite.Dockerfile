FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y ffmpeg curl python3 python3-pip bash-completion git libportaudio2 portaudio19-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

ADD requirements.txt /requirements.txt
RUN python3 -m pip install --break -r /requirements.txt

ENV LD_LIBRARY_PATH=/usr/local/cuda-12/lib64/

ADD record.py /record.py

CMD ["python3", "/record.py"]
