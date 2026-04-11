FROM atmaranto/record:lite
ARG WHISPER_MODEL=Systran/faster-distil-whisper-large-v3
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN python3 -c "from faster_whisper import WhisperModel; model_path = '${WHISPER_MODEL}'; model = WhisperModel(model_path, device='cpu')"
