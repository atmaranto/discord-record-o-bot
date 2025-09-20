import queue
import subprocess
import os
import time
import json

import struct

import scipy.signal
import numpy as np

import discord
from discord.ext import commands
from discord.voice_client import VoiceClient

import threading

import dotenv
dotenv.load_dotenv()

from faster_whisper import WhisperModel

recording_extension = os.environ.get("RECORDING_EXTENSION", ".mp3")

recordings_dir = os.environ.get("RECORDINGS_DIR", "./recordings")
if not os.path.exists(recordings_dir):
    raise ValueError("RECORDINGS_DIR does not exist: " + recordings_dir)

if not os.path.exists(recordings_dir):
    os.makedirs(recordings_dir)

device = os.environ.get("WHISPER_DEVICE", "auto")

model_path = os.environ.get("WHISPER_MODEL", "Systran/faster-distil-whisper-large-v2")
model = WhisperModel(model_path, device=device)

record_o_bot_code = os.environ.get("RECORD_O_BOT_CODE")
if not record_o_bot_code:
    raise ValueError("RECORD_O_BOT_CODE environment variable is not set.")

intents = discord.Intents.default()
intents.members = True

bot = commands.Bot(command_prefix="!" , intents=intents)
who_can_use = os.environ.get("WHO_CAN_USE", None)

if who_can_use is None:
    bot_owner = os.environ.get("BOT_OWNER")
    if not bot_owner:
        raise ValueError("Either WHO_CAN_USE or BOT_OWNER environment variable must be set.")
    
    if not bot_owner.isdigit():
        raise ValueError("BOT_OWNER environment variable is not a valid user ID.")
    bot_owner = int(bot_owner)
    who_can_use = [("user", bot_owner)]
else:
    who_can_use = [item.split(":") for item in who_can_use.split(",")]
    for item in who_can_use:
        if item[0] not in ("user", "role", "guild"):
            raise ValueError("WHO_CAN_USE must be a comma-separated list of user:<id>, role:<id>, or guild:<id>.")
        if not item[1].isdigit():
            raise ValueError("WHO_CAN_USE must be a comma-separated list of user:<id>, role:<id>, or guild:<id>.")
        item[1] = int(item[1])

@bot.check
def check_permissions(ctx):
    if who_can_use is None:
        return True
    
    for item in who_can_use:
        if item[0] == "user" and ctx.user.id == item[1]:
            return True
        if item[0] == "role" and any(role.id == item[1] for role in ctx.user.roles):
            return True
        if item[0] == "guild" and ctx.guild is not None and ctx.guild.id == item[1]:
            return True
    
    return False

voice_channel = None
voice_client = None
save_path = None

class MuxingVoiceClient(VoiceClient):
    def _process_audio_packet(self, data):
        if data.ssrc not in self.user_timestamps:  # First packet from user
            if (
                not self.user_timestamps or not self.sync_start
            ):  # First packet from anyone
                self.first_packet_timestamp = data.receive_time
                silence = 0

            else:  # Previously received a packet from someone else
                silence = (
                    (data.receive_time - self.first_packet_timestamp) * 48000
                ) - 960

        else:  # Previously received a packet from user
            dRT = (
                data.receive_time - self.user_timestamps[data.ssrc][1]
            ) * 48000  # delta receive time
            dT = int(data.receive_time * 48000) - self.user_timestamps[data.ssrc][0]  # delta timestamp
            diff = abs(100 - dT * 100 / dRT)
            if (
                diff > 60 and dT != 960
            ):  # If the difference in change is more than 60% threshold
                silence = dRT - 960
            else:
                silence = dT - 960

        self.user_timestamps.update({data.ssrc: (int(data.receive_time * 48000), data.receive_time)})

        #data.decoded_data = (
        #    struct.pack("<h", 0) * max(0, int(silence)) * opus._OpusStruct.CHANNELS
        #    + data.decoded_data
        #)

        while data.ssrc not in self.ws.ssrc_map:
            time.sleep(0.05)
        self.sink.write(data, self.ws.ssrc_map[data.ssrc]["user_id"], silence)

# Sink class from py-cord
class MuxingSink(discord.sinks.Sink):
    """Combines multiple user streams into one by muxing them, filling in gaps with silence."""
    def __init__(self, path):
        super().__init__()
        self.path = path

        self.user_buffers = {}
        self.keep_last_packets = 4096 * 2 * 2
        self.last_received = 0

        # Keep the last ten seconds of audio together
        self.last_ten_seconds = {}
        self.bytes_written = 0

        self.transcript = {
            "transcription": [],
            "params": {"model": model_path, "language": "en"}
        }
        self.last_save_size = 0
        self.last_save_time = 0

        self.volume_threshold = 0.01

        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "s16le", "-ar", "48000", "-ac", "2", "-i", "pipe:", path],
            stdin=subprocess.PIPE
        )

        self.silence_missing = 0

        self.first_packet_at = time.time()

        self.running = True
        self.queue = queue.Queue()

        self.thread = threading.Thread(target=self.transcribe, daemon=True)
        self.thread.start()
    
    def transcribe(self):
        while self.running:
            item = self.queue.get()
            if item is None:
                break

            user, data, start = item
            start -= self.silence_missing / 48000
            data = scipy.signal.resample(data, int(len(data) * 16000 / 48000))
            data = np.mean(data.reshape(-1, 2), axis=-1)
            print(np.max(np.abs(data)))
            data = data / np.max(np.abs(data)) # Normalize
            result, result_info = model.transcribe(data, vad_filter=True, word_timestamps=True, beam_size=5, language="en")

            result = list(result)

            for seg in result:
                s = {"tokens": []}
                s["start"] = seg.start + start
                s["end"] = seg.end + start
                s["text"] = seg.text
                for word in seg.words:
                    s["tokens"].append({
                        "text": word.word,
                        "timestamps": {
                            "from": "%02d:%02d:%02d,%03d" % ((word.start + start) // 3600, ((word.start + start) // 60) % 60, (word.start + start) % 60, int(((word.start + start) % 1) * 1000)),
                            "to": "%02d:%02d:%02d,%03d" % ((word.end + start) // 3600, ((word.end + start) // 60) % 60, (word.end + start) % 60, int(((word.end + start) % 1) * 1000))
                        }, "speaker": user
                    })
                s["from"] = "%02d:%02d:%02d,%03d" % ((seg.start + start) // 3600, ((seg.start + start) // 60) % 60, (seg.start + start) % 60, int(((seg.start + start) % 1) * 1000))
                s["to"] = "%02d:%02d:%02d,%03d" % ((seg.end + start) // 3600, ((seg.end + start) // 60) % 60, (seg.end + start) % 60, int(((seg.end + start) % 1) * 1000))
                s["speakers"] = [user]
                self.transcript["transcription"].append(s)
            
            if len(self.transcript["transcription"]) - self.last_save_size > 100 or time.time() - self.last_save_time > 300:
                self.save_transcript()

            result = " ".join(item.text for item in result)
            result = result.strip()

            print(user + ":", result)
    
    def write(self, data, user, silence):
        if user not in self.user_buffers:
            self.user_buffers[user] = [0, []]

        if self.user_buffers[user][0] + len(data.decoded_data) > self.keep_last_packets or self.last_received - int(data.receive_time * 48000) > 0.02:
            # Flush the buffer
            self.flush_buffers()
        
        # Write the audio to the buffer
        self.user_buffers[user][1].append(data)
        self.user_buffers[user][0] += len(data.decoded_data) // 2

        # Keep the last ten seconds of audio
        if user not in self.last_ten_seconds:
            self.last_ten_seconds[user] = [0, [], data.receive_time, None]
        
        last_received = self.last_ten_seconds[user][2] or data.receive_time
        time_diff = (data.receive_time - last_received)
        
        arr = np.frombuffer(data.decoded_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.last_ten_seconds[user][1].append(arr)
        self.last_ten_seconds[user][0] += len(data.decoded_data) // 2
        
        if np.max(np.abs(arr)) > self.volume_threshold:
            self.last_ten_seconds[user][2] = data.receive_time
        
        current_time = time.time()
        self.last_ten_seconds[user][3] = self.last_ten_seconds[user][3] or current_time
        
        if (self.last_ten_seconds[user][0] > 10 * 2 * 48000 or time_diff > 0.2) and len(self.last_ten_seconds[user][1]) > 0:
            self.queue.put((getattr(bot.get_user(user), 'display_name', str(user)), np.concatenate(self.last_ten_seconds[user][1]), self.last_ten_seconds[user][3] - self.first_packet_at))
            self.last_ten_seconds[user] = [0, [], 0, 0]
    
    def save_transcript(self):
        with open(self.path + ".json", "w") as f:
            json.dump(self.transcript, f)
            self.last_save_size = len(self.transcript["transcription"])
            self.last_save_time = time.time()
    
    def write_silence(self, silence):
        # Write silence in 100ms chunks
        
        if silence > 0:
            print("Wrote", silence, "samples of silence (" + str(silence / 48000) + "s)")
        silence = int(silence)
        while silence > 0:
            chunk = min(silence, 4800)
            self.p.stdin.write(struct.pack("<h", 0) * chunk * 2)
            silence -= chunk
    
    def flush_buffers(self):
        # Get the user with the most data
        user = max(self.user_buffers, key=lambda user: self.user_buffers[user][0])
        if self.last_received == 0:
            self.last_received = min(int(data.receive_time * 48000) for user in self.user_buffers.keys() for data in self.user_buffers[user][1])
        
        to_t = max(int(data.receive_time * 48000) + (len(data.decoded_data) // 2) for user in self.user_buffers.keys() for data in self.user_buffers[user][1])

        # Mux the other users' data with the user with the most data. Make sure to consider when each user last sent data using int(data.receive_time * 48000).
        full_buffer = np.zeros(max(sum(len(data.decoded_data) // 2 for data in buffer[1]) for buffer in self.user_buffers.values()), dtype=np.int16)
        for u, (size, buffer) in self.user_buffers.items():
            if size == 0: continue

            # Try to guess the offset of the user's data
            offset = max(0, self.last_received - int(buffer[0].receive_time * 48000))
            
            # Align with the end of the buffer to fit all the data in, if necessary
            if offset + size > len(full_buffer):
                #print("Resizing buffer from", len(full_buffer), "to", offset + size)
                offset = len(full_buffer) - size
            
            i = -1
            last_added = 0
            while i+1 < len(buffer):
                i += 1
                offset += last_added
                data = buffer[i]
                #print("Adding", len(data.decoded_data) // 2, "samples from", u, "at offset", offset, "to buffer of size", len(full_buffer))
                last_added = len(data.decoded_data) // 2
                full_buffer[offset:offset+last_added] += np.frombuffer(data.decoded_data, dtype=np.int16)
        
        # Write silence from the last received packet to the current packet
        silence = to_t - self.last_received
        if self.last_received != 0 and silence > 0.2 * 48000:
            self.write_silence(min(silence, 10 * 48000))
            self.silence_missing += silence - min(silence, 10 * 48000)
        self.last_received = max(int(data.receive_time * 48000) for user in self.user_buffers.keys() for data in self.user_buffers[user][1])

        # Clear the buffers
        for user in self.user_buffers:
            self.user_buffers[user] = [0, []]

        # Write to the process
        data = full_buffer.tobytes()
        self.p.stdin.write(data)
        self.bytes_written += len(data)
    
    def cleanup(self):
        print("Cleaning up (fake)")
    
    def real_cleanup(self):
        super().cleanup()
        print("Cleaning up (real)")
        self.flush_buffers()

        self.running = False
        self.queue.put(None)
        self.thread.join()

        self.save_transcript()

        self.p.communicate()

@bot.slash_command(name="record", description="Records a message.")
async def record(ctx: discord.Interaction, path: str):
    global voice_channel, voice_client, save_path
    
    if getattr(ctx.user, "voice") is None or ctx.user.voice.channel is None:
        await ctx.response.send_message("You are not in a voice channel.")
        return
    
    if voice_channel is not None:
        await ctx.response.send_message("The bot is already recording. Please stop the current recording in \"" + voice_channel.name + "\" before starting a new one.")
        return

    # Strip all non-alphanumeric characters from the path
    path = "".join([c for c in path if (c.isalnum() and c.isascii()) or c in "._-"])[:50] + recording_extension
    if os.path.exists(os.path.join(recordings_dir, path)):
        await ctx.response.send_message("A file with that name already exists.")
        return

    await ctx.response.send_message("Recording...")

    voice_client = await ctx.user.voice.channel.connect(cls=MuxingVoiceClient)
    print("Created")
    voice_channel = ctx.user.voice.channel
    save_path = os.path.join(recordings_dir, path)
    sink = MuxingSink(save_path)

    async def on_stop(s):
        global voice_channel, voice_client, save_path
        if voice_channel is not None:
            print("WARNING! Possible attempt to stop recording early?")

            voice_client = await voice_channel.connect(cls=MuxingVoiceClient)
            voice_client.start_recording(sink, on_stop, sync_start=False)
            return

        sink.real_cleanup()

        voice_channel = None
        voice_client = None
        save_path = None

    voice_client.start_recording(sink, on_stop, sync_start=False)

    await ctx.response.edit_message(content="Recording to " + path + ".")

@bot.slash_command(name="stop", description="Stops recording.")
async def stop(ctx: discord.Interaction):
    global voice_channel, voice_client, save_path
    if voice_channel is None:
        await ctx.response.send_message("The bot is not recording.")
        return

    await ctx.response.send_message("Stopping...")

    voice_channel = None
    voice_client.stop_recording()
    await voice_client.disconnect()
    voice_client = None
    save_path = None

    await ctx.edit(content="Stopped recording.")

@bot.slash_command(name="ping", description="Pings the bot.")
async def ping(ctx: discord.Interaction):
    await ctx.response.send_message("Pong!")

@bot.event
async def on_ready():
    print("Ready")

bot.run(record_o_bot_code)
