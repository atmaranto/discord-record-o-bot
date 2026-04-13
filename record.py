import asyncio
import queue
import subprocess
import os
import time
import json

import struct
import traceback

import numpy as np
try:
    from faster_whisper import WhisperModel

    import scipy.signal
except ImportError:
    TRANSCRIPTION_AVAILABLE = False
    print("NOTE: Whisper transcription is not available. To enable it, install the faster-whisper, scipy, and numpy packages.")
else:
    TRANSCRIPTION_AVAILABLE = True

import discord
from discord.ext import commands
from discord.voice.client import VoiceClient

import threading

import dotenv
dotenv.load_dotenv()

# import logging
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("discord.voice.receive.reader").setLevel(logging.DEBUG)

recording_extension = os.environ.get("RECORDING_EXTENSION", ".mp3")
individual_stream_extension = os.environ.get("INDIVIDUAL_STREAM_EXTENSION", "mp3")

recordings_dir = os.environ.get("RECORDINGS_DIR", "./recordings")
if __name__ == "__main__":
    if not os.path.exists(recordings_dir):
        raise ValueError("RECORDINGS_DIR does not exist: " + recordings_dir)

    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)

if TRANSCRIPTION_AVAILABLE:
    device = os.environ.get("WHISPER_DEVICE", "auto")

    model_path = os.environ.get("WHISPER_MODEL", "Systran/faster-distil-whisper-large-v2")
    model = WhisperModel(model_path, device=device)

record_o_bot_code = os.environ.get("RECORD_O_BOT_CODE")
if not record_o_bot_code and __name__ == "__main__":
    raise ValueError("RECORD_O_BOT_CODE environment variable is not set.")

intents = discord.Intents.all()
intents.members = True

bot = commands.Bot(command_prefix="!" , intents=intents)
who_can_use = None

if __name__ == "__main__":
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
_keepalive_task = None

class CompatPacket:
    """Wraps VoiceData to provide backward-compatible attributes for sink processing."""
    __slots__ = ('decoded_data', 'receive_time')
    def __init__(self, pcm, receive_time):
        self.decoded_data = pcm
        self.receive_time = receive_time


def _rtp_to_time(rtp_anchors, ssrc, rtp_timestamp):
    """Convert an RTP timestamp to a normalized wall-clock time using per-SSRC anchoring.

    On the first packet for each SSRC, the current wall-clock time is recorded
    as an anchor. Subsequent packets derive their time from the RTP timestamp
    delta, giving accurate relative timing even when packets arrive in batches.
    """
    if ssrc not in rtp_anchors:
        wall = time.time()
        rtp_anchors[ssrc] = (rtp_timestamp, wall)
        return wall
    anchor_rtp, anchor_wall = rtp_anchors[ssrc]
    # Handle uint32 wrapping
    delta = (rtp_timestamp - anchor_rtp) & 0xFFFFFFFF
    return anchor_wall + delta / 48000.0


def _calculate_silence(user_timestamps, first_packet_timestamp, ssrc, receive_time):
    """Calculate silence (in samples) between the current packet and the previous one for a given SSRC."""
    if ssrc not in user_timestamps:
        if not user_timestamps or first_packet_timestamp is None:
            return 0, receive_time  # first packet from anyone
        else:
            return ((receive_time - first_packet_timestamp) * 48000) - 960, first_packet_timestamp
    else:
        dRT = (receive_time - user_timestamps[ssrc][1]) * 48000
        dT = int(receive_time * 48000) - user_timestamps[ssrc][0]
        if dRT > 0:
            diff = abs(100 - dT * 100 / dRT)
        else:
            diff = 0
        if diff > 60 and dT != 960:
            return dRT - 960, first_packet_timestamp
        else:
            return dT - 960, first_packet_timestamp

class TranscribingSink(discord.sinks.Sink):
    """Base class providing shared transcription state and worker thread for both sink types."""

    def __init__(self, transcribe=TRANSCRIPTION_AVAILABLE):
        super().__init__()
        self.first_packet_at = None
        self._rtp_anchors = {}
        self.silence_missing = 0
        self.last_ten_seconds = {}
        self.volume_threshold = 0.01

        self.transcription_enabled = transcribe and TRANSCRIPTION_AVAILABLE
        if self.transcription_enabled:
            self.transcript = {
                "transcription": [],
                "params": {"model": model_path, "language": "en"}
            }
        self.last_save_size = 0
        self.last_save_time = 0

        self.running = True
        self.queue = queue.Queue()

        self.thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self.thread.start()

    def _transcribe_loop(self):
        if not TRANSCRIPTION_AVAILABLE or not self.transcription_enabled:
            return

        while self.running:
            item = self.queue.get()
            if item is None:
                break

            user, data, start = item
            start -= self.silence_missing / 48000
            data = scipy.signal.resample(data, int(len(data) * 16000 / 48000))
            data = np.mean(data.reshape(-1, 2), axis=-1)
            data = data / np.max(np.abs(data))
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

    def save_transcript(self):
        raise NotImplementedError


# Sink class from py-cord
class MuxingSink(TranscribingSink):
    """Combines multiple user streams into one by muxing them, filling in gaps with silence."""
    def __init__(self, path, transcribe=TRANSCRIPTION_AVAILABLE):
        self.path = path

        self.user_buffers = {}
        self.keep_last_packets = 4096 * 2 * 2
        self.last_received = 0
        self.bytes_written = 0

        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "s16le", "-ar", "48000", "-ac", "2", "-i", "pipe:", path],
            stdin=subprocess.PIPE
        )

        super().__init__(transcribe=transcribe)
    
    def write(self, data, user):
        receive_time = _rtp_to_time(self._rtp_anchors, data.packet.ssrc, data.packet.timestamp)
        if self.first_packet_at is None:
            self.first_packet_at = receive_time
        user_id = user.id if user is not None else data.packet.ssrc
        packet = CompatPacket(data.pcm, receive_time)
        # print("Writing packet from", user_id, "with", len(packet.decoded_data), "bytes")
        if user_id not in self.user_buffers:
            self.user_buffers[user_id] = [0, []]

        if self.user_buffers[user_id][0] + len(packet.decoded_data) > self.keep_last_packets or self.last_received - int(packet.receive_time * 48000) > 0.02:
            # Flush the buffer
            self.flush_buffers()
        
        # Write the audio to the buffer
        self.user_buffers[user_id][1].append(packet)
        self.user_buffers[user_id][0] += len(packet.decoded_data) // 2

        # Keep the last ten seconds of audio (for transcription)
        if not self.transcription_enabled:
            return

        if user_id not in self.last_ten_seconds:
            self.last_ten_seconds[user_id] = [0, [], receive_time, None]
        
        last_received = self.last_ten_seconds[user_id][2] or receive_time
        time_diff = (receive_time - last_received)
        
        arr = np.frombuffer(packet.decoded_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.last_ten_seconds[user_id][1].append(arr)
        self.last_ten_seconds[user_id][0] += len(packet.decoded_data) // 2
        
        if np.max(np.abs(arr)) > self.volume_threshold:
            self.last_ten_seconds[user_id][2] = receive_time
        
        self.last_ten_seconds[user_id][3] = self.last_ten_seconds[user_id][3] or receive_time
        
        if (self.last_ten_seconds[user_id][0] > 10 * 2 * 48000 or time_diff > 0.2) and len(self.last_ten_seconds[user_id][1]) > 0:
            display_name = getattr(user, 'display_name', str(user_id))
            self.queue.put((display_name, np.concatenate(self.last_ten_seconds[user_id][1]), self.last_ten_seconds[user_id][3] - self.first_packet_at))
            self.last_ten_seconds[user_id] = [0, [], 0, 0]
    
    def save_transcript(self):
        if not TRANSCRIPTION_AVAILABLE or not self.transcription_enabled:
            return
        
        with open(self.path + ".json", "w") as f:
            json.dump(self.transcript, f)
            self.last_save_size = len(self.transcript["transcription"])
            self.last_save_time = time.time()
    
    def write_silence(self, silence):
        # Write silence in 100ms chunks
        
        # if silence > 0:
        #     print("Wrote", silence, "samples of silence (" + str(silence / 48000) + "s)")
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
        self.flush_buffers()

        self.running = False
        self.queue.put(None)

        async def _clean():
            await self.vc.loop.run_in_executor(None, self.thread.join)

            self.save_transcript()

            await self.vc.loop.run_in_executor(None, self.p.communicate)

        self.vc.loop.create_task(_clean())

class MultiStreamSink(TranscribingSink):
    """Saves each user's audio to a separate file in a recording directory, with optional transcription."""
    def __init__(self, dir_path, individual_ext=individual_stream_extension, transcribe=TRANSCRIPTION_AVAILABLE):
        self.dir_path = dir_path
        self.individual_ext = individual_ext
        os.makedirs(dir_path, exist_ok=True)

        self.user_processes = {}
        self._user_timestamps = {}
        self._first_packet_timestamp = None

        super().__init__(transcribe=transcribe)

    def _get_user_process(self, user):
        if user not in self.user_processes:
            path = os.path.join(self.dir_path, str(user) + "." + self.individual_ext)
            self.user_processes[user] = subprocess.Popen(
                ["ffmpeg", "-y", "-f", "s16le", "-ar", "48000", "-ac", "2", "-i", "pipe:", path],
                stdin=subprocess.PIPE
            )
        return self.user_processes[user]

    def save_transcript(self):
        if not TRANSCRIPTION_AVAILABLE or not self.transcription_enabled:
            return

        with open(os.path.join(self.dir_path, "transcript.json"), "w") as f:
            json.dump(self.transcript, f)
            self.last_save_size = len(self.transcript["transcription"])
            self.last_save_time = time.time()

    def write_silence_for_user(self, user, silence_samples):
        p = self._get_user_process(user)
        while silence_samples > 0:
            chunk = min(silence_samples, 4800)
            p.stdin.write(struct.pack("<h", 0) * chunk * 2)
            silence_samples -= chunk

    def write(self, data, user):
        receive_time = _rtp_to_time(self._rtp_anchors, data.packet.ssrc, data.packet.timestamp)
        if self.first_packet_at is None:
            self.first_packet_at = receive_time
        user_id = user.id if user is not None else data.packet.ssrc
        ssrc = data.packet.ssrc

        # Calculate silence from timing
        silence, self._first_packet_timestamp = _calculate_silence(
            self._user_timestamps, self._first_packet_timestamp, ssrc, receive_time
        )
        self._user_timestamps[ssrc] = (int(receive_time * 48000), receive_time)

        p = self._get_user_process(user_id)

        # Write silence for this user's individual stream
        silence_samples = int(max(0, silence))
        if silence_samples > 0:
            capped = min(silence_samples, 10 * 48000)
            self.write_silence_for_user(user_id, capped)
            if silence_samples > capped:
                self.silence_missing += silence_samples - capped

        # Write audio data to this user's ffmpeg process
        p.stdin.write(data.pcm)

        # Transcription handling
        if not self.transcription_enabled:
            return

        if user_id not in self.last_ten_seconds:
            self.last_ten_seconds[user_id] = [0, [], receive_time, None]

        last_received = self.last_ten_seconds[user_id][2] or receive_time
        time_diff = (receive_time - last_received)

        arr = np.frombuffer(data.pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self.last_ten_seconds[user_id][1].append(arr)
        self.last_ten_seconds[user_id][0] += len(data.pcm) // 2

        if np.max(np.abs(arr)) > self.volume_threshold:
            self.last_ten_seconds[user_id][2] = receive_time

        self.last_ten_seconds[user_id][3] = self.last_ten_seconds[user_id][3] or receive_time

        if (self.last_ten_seconds[user_id][0] > 10 * 2 * 48000 or time_diff > 0.2) and len(self.last_ten_seconds[user_id][1]) > 0:
            display_name = getattr(user, 'display_name', str(user_id))
            self.queue.put((display_name, np.concatenate(self.last_ten_seconds[user_id][1]), self.last_ten_seconds[user_id][3] - self.first_packet_at))
            self.last_ten_seconds[user_id] = [0, [], 0, 0]

    def cleanup(self):
        super().cleanup()
        self.running = False
        self.queue.put(None)

        async def _clean():
            await self.vc.loop.run_in_executor(None, self.thread.join)

            self.save_transcript()

            for p in self.user_processes.values():
                p.stdin.close()
                await self.vc.loop.run_in_executor(None, p.wait)

            self._generate_consolidate_scripts()
        
        self.vc.loop.create_task(_clean())

    def _generate_consolidate_scripts(self):
        ext = self.individual_ext

        # --- Consolidate.py ---
        py_content = '''#!/usr/bin/env python3
"""Consolidate individual audio streams into a single file.

Reads the transcription transcript.json to identify speakers and timing,
then uses ffmpeg to combine all individual streams.

Usage:
    python Consolidate.py [output_file] [--mix]

Options:
    output_file  Output filename (default: consolidated.EXT)
    --mix        Mix all streams into a single audio track.
                 Without --mix, each user is kept as a separate audio
                 track in a multi-stream container (e.g. .mkv).
"""
import json
import os
import subprocess
import sys

AUDIO_EXT = "__AUDIO_EXT__"


def find_transcript():
    """Load transcript.json if it exists."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcript.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def find_audio_files(exclude=None):
    """Find all individual stream audio files in this directory."""
    exclude = set(exclude or [])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files = []
    for f in sorted(os.listdir(script_dir)):
        full = os.path.join(script_dir, f)
        if os.path.isfile(full) and f.endswith("." + AUDIO_EXT):
            if f not in exclude and not f.startswith("consolidated"):
                files.append(f)
    return files


def show_transcript_info(transcript):
    """Display speaker info from the transcript."""
    if not transcript or "transcription" not in transcript:
        return
    speakers = {}
    for seg in transcript["transcription"]:
        for speaker in seg.get("speakers", []):
            if speaker not in speakers:
                speakers[speaker] = {"segments": 0, "duration": 0.0}
            speakers[speaker]["segments"] += 1
            speakers[speaker]["duration"] += seg.get("end", 0) - seg.get("start", 0)
    print(f"Transcript: {len(speakers)} speaker(s)")
    for speaker, info in speakers.items():
        segs = info["segments"]
        dur = info["duration"]
        print(f"  {speaker}: {segs} segment(s), {dur:.1f}s speaking time")


def main():
    mix_mode = "--mix" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--mix"]

    if mix_mode:
        default_output = f"consolidated.{AUDIO_EXT}"
    else:
        default_output = "consolidated.mkv"
    output = args[0] if args else default_output

    audio_files = find_audio_files(exclude={output})
    if not audio_files:
        print("No audio files found to consolidate.")
        sys.exit(1)

    transcript = find_transcript()

    n = len(audio_files)
    print(f"Found {n} audio stream(s):")
    for f in audio_files:
        print(f"  - {f}")

    if transcript:
        show_transcript_info(transcript)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = ["ffmpeg"]
    for f in audio_files:
        cmd.extend(["-i", os.path.join(script_dir, f)])

    if mix_mode:
        cmd.extend([
            "-filter_complex",
            f"amix=inputs={n}:duration=longest:dropout_transition=0",
        ])
    else:
        for i in range(n):
            cmd.extend(["-map", f"{i}:a"])

    cmd.extend(["-y", os.path.join(script_dir, output)])

    print(f"\\nConsolidating {n} stream(s) into {output}...")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"Successfully created {output}")
    else:
        print(f"Error: ffmpeg exited with code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''.replace("__AUDIO_EXT__", ext)

        # --- Consolidate.bat ---
        bat_content = '@echo off\n'
        bat_content += 'REM Consolidate individual audio streams into a single file\n'
        bat_content += 'REM Usage: Consolidate.bat [output_file] [--mix]\n'
        bat_content += 'python "%~dp0Consolidate.py" %*\n'

        # --- Consolidate.sh ---
        sh_content = '#!/bin/bash\n'
        sh_content += '# Consolidate individual audio streams into a single file\n'
        sh_content += '# Usage: ./Consolidate.sh [output_file] [--mix]\n'
        sh_content += 'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        sh_content += 'python3 "$SCRIPT_DIR/Consolidate.py" "$@"\n'

        for filename, content in [("Consolidate.py", py_content), ("Consolidate.bat", bat_content), ("Consolidate.sh", sh_content)]:
            with open(os.path.join(self.dir_path, filename), "w", newline=('\r\n' if filename.endswith('.bat') else '\n')) as f:
                f.write(content)

@bot.slash_command(name="record", description="Records a message.")
async def record(ctx: discord.Interaction, path: str, transcribe: bool = TRANSCRIPTION_AVAILABLE, mode: str = "muxed", individual_format: str = individual_stream_extension):
    global voice_channel, voice_client, save_path
    
    if getattr(ctx.user, "voice") is None or ctx.user.voice.channel is None:
        await ctx.response.send_message("You are not in a voice channel.")
        return
    
    if voice_channel is not None:
        await ctx.response.send_message("The bot is already recording. Please stop the current recording in \"" + voice_channel.name + "\" before starting a new one.")
        return

    if mode not in ("muxed", "multi"):
        await ctx.response.send_message("Mode must be 'muxed' or 'multi'.")
        return

    # Strip all non-alphanumeric characters from the path
    sanitized = "".join([c for c in path if (c.isalnum() and c.isascii()) or c in "._-"])[:50]
    if mode == "multi":
        path = sanitized + "_recording"
    else:
        path = sanitized + recording_extension
    if os.path.exists(os.path.join(recordings_dir, path)):
        await ctx.response.send_message("A file or directory with that name already exists.")
        return
    
    if transcribe and not TRANSCRIPTION_AVAILABLE:
        await ctx.response.send_message("Transcription is not available on this bot.")
        return

    message = await ctx.response.send_message("Connecting to voice...")

    voice_client = await ctx.user.voice.channel.connect()
    print("Created")
    voice_channel = ctx.user.voice.channel
    save_path = os.path.join(recordings_dir, path)
    if mode == "multi":
        sink = MultiStreamSink(save_path, individual_ext=individual_format, transcribe=transcribe)
    else:
        sink = MuxingSink(save_path, transcribe=transcribe)

    def on_stop(s):
        global voice_channel, voice_client, save_path

        voice_channel = None
        voice_client = None
        save_path = None

    voice_client.start_recording(sink, on_stop)

    # Keep the event loop responsive during recording.
    # Pycord's SocketReader dispatches packet-processing tasks via
    # loop.call_soon() from a background thread, which does NOT wake
    # the Windows ProactorEventLoop from its IOCP poll.  Without a
    # periodic timer, packets silently accumulate in the ready queue
    # and the jitter buffer overflows when they are finally processed
    # in a single burst.
    async def _event_loop_keepalive():
        while voice_client is not None:
            await asyncio.sleep(0.02)

    global _keepalive_task
    _keepalive_task = bot.loop.create_task(_event_loop_keepalive())

    await message.edit(content="Recording to " + path + ".")

@bot.slash_command(name="stop", description="Stops recording.")
async def stop(ctx: discord.Interaction):
    global voice_channel, voice_client, save_path
    if voice_channel is None:
        await ctx.response.send_message("The bot is not recording.")
        return

    await ctx.response.send_message("Stopping...")

    global _keepalive_task
    if _keepalive_task is not None:
        _keepalive_task.cancel()
        _keepalive_task = None

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

if __name__ == "__main__":
    bot.run(record_o_bot_code)
