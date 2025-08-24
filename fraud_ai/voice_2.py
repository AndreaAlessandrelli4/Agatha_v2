import asyncio
import nltk
import aiohttp
import numpy as np
import sounddevice as sd
from nltk.tokenize import sent_tokenize
from openai import AsyncOpenAI
from fraud_ai.config import OPENAI_API_KEY, ELEVEN_KEY

# ===========================
# CONFIG
# ===========================
nltk.download("punkt", quiet=True)

ELEVENLABS_API_KEY = ELEVEN_KEY
OPENAI_KEY = OPENAI_API_KEY
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Replace with your ElevenLabs voice ID

SAMPLERATE = 22050
CHANNELS = 1
DTYPE = 'int16'

client = AsyncOpenAI(api_key=OPENAI_KEY)

# ===========================
# ElevenLabs Streaming
# ===========================
async def elevenlabs_stream_tts(text: str):
    """
    Stream raw PCM (16-bit signed, mono, 22050Hz) from ElevenLabs
    and yield chunks aligned to full int16 frames.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream?output_format=pcm_22050"

    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.35,
            "similarity_boost": 0.75
        }
    }

    headers = {
        "Accept": "audio/wav",
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                raise RuntimeError(f"ElevenLabs TTS failed [{resp.status}]: {err_text}")

            leftover = b""
            async for net_chunk in resp.content.iter_chunked(4096):
                if not net_chunk:
                    continue

                data = leftover + net_chunk
                # Align to full int16 frames (2 bytes each sample)
                frame_count = len(data) // 2
                full_bytes = data[:frame_count * 2]
                leftover = data[frame_count * 2:]

                if full_bytes:
                    yield full_bytes

            # Flush any leftover full frames at the end
            if leftover:
                if len(leftover) % 2 != 0:
                    leftover = leftover[:-1]  # drop stray byte
                if leftover:
                    yield leftover

# ===========================
# TTS Playback Worker
# ===========================
async def tts_worker(tts_queue: asyncio.Queue):
    """Continuously takes text from queue, streams from ElevenLabs, and plays it."""
    with sd.OutputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype=DTYPE) as stream:
        while True:
            text = await tts_queue.get()
            if text is None:
                break
            async for pcm_chunk in elevenlabs_stream_tts(text):
                # Safety trim in case of unexpected odd-length chunk
                if len(pcm_chunk) % 2 != 0:
                    pcm_chunk = pcm_chunk[:-1]
                audio_data = np.frombuffer(pcm_chunk, dtype=np.int16)
                stream.write(audio_data)

# ===========================
# GPT Chat + Sentence TTS
# ===========================
async def chat_and_speak_live(user_input):
    """
    Stream GPT answer, send each finished sentence to TTS in real-time.
    """
    tts_queue = asyncio.Queue()
    tts_task = asyncio.create_task(tts_worker(tts_queue))

    buffer = ""
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_input}],
        stream=True
    )

    async for event in stream:
        delta = event.choices[0].delta
        if delta and delta.content:
            token = delta.content
            print(token, end="", flush=True)
            buffer += token

            # Sentence splitting
            sentences = sent_tokenize(buffer)
            if len(sentences) > 1:
                for s in sentences[:-1]:
                    await tts_queue.put(s.strip())
                buffer = sentences[-1]

    if buffer.strip():
        await tts_queue.put(buffer.strip())

    await tts_queue.put(None)
    await tts_task
    print()

# ===========================
# Main loop
# ===========================
async def main():
    print("💬 Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("👋 Goodbye!")
            break
        try:
            await chat_and_speak_live(user_input)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())