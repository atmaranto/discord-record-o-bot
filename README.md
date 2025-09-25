# Project Description
Have you ever wanted to record and transcribe your Discord conversations, but bots that could do that seemed just a bit too rare or untrustworthy?
Now you can run your own!

Record-o-Bot is an extremely simple record-and-transcribe bot. That's it!

## Configuration

Record-o-Bot uses the `dotenv` library to load extra environmental variables from a `.env` file. This is recommended for securely passing secrets to the bot.

The following environmental variables are supported:

```bash
RECORD_O_BOT_CODE=The Discord token for this bot
RECORDINGS_DIR=./recordings # The directory to store recordings in (defaults to $PWD/recordings)
BOT_OWNER=The Discord user ID of the bot owner
WHO_CAN_USE=guild:the ID of a guild where everyone can use a bot,role:the ID of a role where everyone can use the bot,user:The ID of a user who can use it
WHISPER_DEVICE=auto, cpu, or cuda # The device for Whisper to prefer
WHISPER_MODEL=Systran/faster-distil-whisper-large-v2 # The model for Whisper to use
RECORDING_EXTENSION=.mp3 # The extension to save recordings as
```

One of `WHO_CAN_USE` and `BOT_OWNER` must be specified, but `WHO_CAN_USE` is preferred over `BOT_OWNER` is present.

### Discord IDs

To get Discord IDs, make sure "Developer Mode" is enabled in your Discord client. Then, right click on any Discord entity (a role, a user, a server), and click Copy X ID.

## Usage (Command line)

Simply create a venv and install the requirements, then run it. Please make sure you have CUDA installed and available if you want to take advantage of GPU accelaration,
and ensure that you have `ffmpeg` in your path.

```bash
python -m pip install -r requirements.txt

python record.py
```

Simply use `/record <filename>` in Discord to start recording to `<filename>.mp3` and transcribing to a `<filename>.json`.

## Usage (Docker)
This bot is fully compatible with Docker, and in fact, this is the preferred method for running Record-o-Bot. It is available at the following tags on Docker Hub:

```bash
atmaranto/record:latest           # The latest full image, including the predownloaded transcription model.

atmaranto/record:lite             # The latest image without the transcription model. This will be downloaded
                                  # automatically, but you can mount /root/.cache/huggingface to share your
                                  # host's huggingface directory.

atmaranto/record:no-transcription # The latest image with no transcription capabilities whatsoever. Transcription
                                  # is disabled, and the model will not be downloaded on first run.
```

To run Record-o-Bot with Docker, simply use the following command:

```bash
docker run -it --gpus all --name record-bot -v /path/to/recordings:/recordings -v /path/to/dotenv/file:/.env atmaranto/record:latest # or lite or no-transcription
```

`record.py` is located at `/record.py` in the container, so you can replace it with a volume mount if you wish to run with a modified version of the script.
