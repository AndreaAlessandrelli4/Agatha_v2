import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile
import time
import webrtcvad
import openai
from elevenlabs import ElevenLabs
from fraud_ai.config import ELEVEN_KEY, OPENAI_API_KEY


openai.api_key = OPENAI_API_KEY  # <-- metti la tua chiave qui
eleven_client = ElevenLabs(api_key=ELEVEN_KEY)  # <-- metti la tua chiave qui


def transcribe_with_elevenlabs(temp_path):
    with open(temp_path, "rb") as f:
        result_stream = eleven_client.speech_to_text.convert(
            model_id="scribe_v1",#scribe_v1_experimental
            file=f
        )

        texts = []
        lang_code = None
        lang_prob = None

        for key, value in result_stream:
            if key == "language_code":
                lang_code = value
            elif key == "language_probability":
                lang_prob = value
            elif key == "text":
                texts.append(value)

    final_text = " ".join(t.strip() for t in texts if t.strip())
    return final_text.strip(), lang_code, lang_prob

def listen_and_transcribe(
    stt_enabled=True,
    stt_provider="openai",
    silence_duration=0.5,
    max_duration=10,
    fs=16000,
    vad_aggressiveness=2,
    language=None
):
    # === Text input fallback ===
    if not stt_enabled:
        manual_text = input("Customer: ").strip()
        return manual_text if manual_text else "silence"

    vad = webrtcvad.Vad(vad_aggressiveness)
    buffer = []
    start_time = time.time()
    speech_started = False
    last_speech = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal speech_started, last_speech
        audio_pcm = (indata[:, 0] * 32768).astype(np.int16).tobytes()
        if vad.is_speech(audio_pcm, fs):
            speech_started = True
            last_speech = time.time()
        buffer.append(indata.copy())

    with sd.InputStream(callback=callback, channels=1, samplerate=fs, blocksize=int(fs * 0.03)):
        while time.time() - start_time < max_duration:
            if speech_started and time.time() - last_speech > silence_duration:
                break
            sd.sleep(50)

    audio = np.concatenate(buffer, axis=0)
    if not speech_started or len(audio) < fs * 0.2:
        return "silence"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, fs)
        temp_path = f.name

    # === STT processing ===
    if stt_provider.lower() == "openai":
        with open(temp_path, "rb") as f:
            resp = openai.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f,
                language=language
            )
        transcription_text = resp.text.strip()
        return transcription_text if transcription_text else "silence"

    elif stt_provider.lower() == "elevenlabs":
        transcription_text, detected_lang, lang_conf = transcribe_with_elevenlabs(temp_path)
        transcription_text = transcription_text.strip()
        return transcription_text if transcription_text else "silence"

    else:
        raise ValueError("stt_provider must be 'openai' or 'elevenlabs'.")