FROM atmaranto/record:lite
RUN python3 -c "from faster_whisper import WhisperModel; model_path = 'Systran/faster-distil-whisper-large-v2'; model = WhisperModel(model_path, device='cpu')"
