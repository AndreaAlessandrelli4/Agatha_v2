import os
import threading
import time
import asyncio
import streamlit as st
from dotenv import load_dotenv
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from st_aggrid import StAggridTheme
import pandas as pd
import numpy as np
import webrtcvad
import av
import tempfile
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_autorefresh import st_autorefresh

# Fraud AI imports
from fraud_ai.data import get_db, update_transaction
from fraud_ai.alerts import get_alerts, update_alert
from fraud_ai.models import Transaction
from fraud_ai.whitelist import add_to_whitelist, is_card_whitelisted, remove_from_whitelist
from fraud_ai.blocked import add_to_blocked, is_card_blocked, remove_from_blocked
from fraud_ai.reset_password import add_password_reset, has_password_reset, remove_password_reset
from fraud_ai.conversation import get_conversation

# Local modules
from demo_runner import prepare_demo_db, run_fraud_flow_demo
from style_CSS import page_style, chat_style, tab_style, bottom_style

load_dotenv()

class AudioProcessor:
    def __init__(self, vad_aggressiveness=2, silence_duration=0.8, max_record_seconds=10, sample_rate=16000):
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.silence_duration = silence_duration
        self.max_record_seconds = max_record_seconds
        self.sample_rate = sample_rate
        self.audio_buffer = []
        self.speech_started = False
        self.last_speech_time = None
        self.start_time = None
        self.transcription = None
        self.lock = threading.Lock()
        self.stopped = False

    def recv(self, frame: av.AudioFrame):
        if self.stopped:
            return None
        audio = frame.to_ndarray(format="s16", layout="mono")
        pcm_bytes = audio.tobytes()
        frame_duration_ms = 30
        frame_length = int(self.sample_rate * frame_duration_ms / 1000) * 2
        for i in range(0, len(pcm_bytes), frame_length):
            chunk = pcm_bytes[i:i+frame_length]
            if len(chunk) < frame_length:
                break
            is_speech = self.vad.is_speech(chunk, self.sample_rate)
            now = time.time()
            if self.start_time is None:
                self.start_time = now
            if is_speech:
                self.speech_started = True
                self.last_speech_time = now
                self.audio_buffer.append(chunk)
            else:
                if self.speech_started and (now - self.last_speech_time > self.silence_duration):
                    self.process_audio()
                    self.reset()
            if now - self.start_time > self.max_record_seconds:
                if self.speech_started:
                    self.process_audio()
                self.reset()
        return frame

    def process_audio(self):
        if not self.audio_buffer:
            return
        audio_bytes = b"".join(self.audio_buffer)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.frombuffer(audio_bytes, dtype=np.int16), self.sample_rate)
            temp_path = f.name
        def transcribe():
            try:
                text = transcribe_audio_file(temp_path)
                with self.lock:
                    self.transcription = text
            except Exception as e:
                with self.lock:
                    self.transcription = f"Error: {e}"
        threading.Thread(target=transcribe, daemon=True).start()

    def reset(self):
        self.audio_buffer = []
        self.speech_started = False
        self.last_speech_time = None
        self.start_time = None

    def stop(self):
        self.stopped = True
        self.reset()

def transcribe_audio_file(file_path, stt_provider="openai", language=None):
    import openai
    from elevenlabs import ElevenLabs
    openai.api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    eleven_api_key = st.session_state.get("eleven_api_key") or os.getenv("ELEVEN_API_KEY")
    eleven_client = ElevenLabs(api_key=eleven_api_key)
    if stt_provider.lower() == "openai":
        with open(file_path, "rb") as f:
            resp = openai.Audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f,
                language=language
            )
        return resp.text.strip()
    elif stt_provider.lower() == "elevenlabs":
        with open(file_path, "rb") as f:
            result_stream = eleven_client.speech_to_text.convert(
                model_id="scribe_v1",
                file=f
            )
            texts = [v for k,v in result_stream if k=="text"]
        return " ".join(texts).strip()
    else:
        raise ValueError("Unsupported STT provider")

def run_demo_thread(tts_backend, stt_enabled, stt_provider, name, surname, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    db, alert, last_tx, recent_txs = prepare_demo_db(name, surname)
    asyncio.run(run_fraud_flow_demo(
        db, alert, last_tx, recent_txs,
        tts_backend=tts_backend,
        stt_enabled=stt_enabled,
        stt_provider=stt_provider,
    ))
    with open("demo_done.flag", "w") as f:
        f.write("done")

st.set_page_config(page_title="Agatha The AI Fraud Analyst", page_icon="🕵️", layout="wide")

for key, default in {
    "name_input_option": "John",
    "surname_input_option": "Doe",
    "demo_started": False,
    "end": False,
    "start_demo_triggered": False,
    "alert_index": 0,
    "analyst_notes": "",
    "refresh_counter": 0,
    "chat_messages": [],
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "eleven_api_key": os.getenv("ELEVEN_API_KEY"),
    "audio_processor": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.container():
    st.markdown("<h3>Personalize the demo</h3>", unsafe_allow_html=True)
    name_input = st.text_input("Your Name", value=st.session_state.name_input_option)
    surname_input = st.text_input("Your Surname", value=st.session_state.surname_input_option)
    start_demo_clicked = st.button("Start Demo")

if start_demo_clicked and not st.session_state.demo_started:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.session_state.name_input_option = name_input or "John"
    st.session_state.surname_input_option = surname_input or "Doe"
    st.session_state.demo_started = True
    st.session_state.end = False
    st.session_state.start_demo_triggered = False
    st.session_state.alert_index = 0
    st.session_state.analyst_notes = ""
    st.session_state.chat_messages = []
    st.session_state.refresh_counter = 0
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY")
    st.session_state.eleven_api_key = os.getenv("ELEVEN_API_KEY")
    st.session_state.audio_processor = AudioProcessor()
    if os.path.exists("demo_done.flag"):
        os.remove("demo_done.flag")
    st.rerun()

# Auto-refresh every 1 second to detect demo end flag
if st.session_state.demo_started and not st.session_state.end:
    st_autorefresh(interval=1000, key="demo_autorefresh")

if st.session_state.demo_started:
    st.markdown(f"### Welcome {st.session_state.name_input_option} {st.session_state.surname_input_option}!")
    st.markdown("### Dashboard")

    if not st.session_state.start_demo_triggered:
        st.session_state.start_demo_triggered = True
        name = st.session_state.name_input_option
        surname = st.session_state.surname_input_option
        api_key = st.session_state.openai_api_key or os.getenv("OPENAI_API_KEY")
        def delayed_start(name, surname, api_key):
            time.sleep(1)
            run_demo_thread(
                tts_backend="openai",
                stt_enabled=True,
                stt_provider="openai",
                name=name,
                surname=surname,
                api_key=api_key,
            )
        threading.Thread(target=delayed_start, args=(name, surname, api_key), daemon=True).start()

    # Detect demo end and stop recorder + rerun immediately
    if not st.session_state.end and os.path.exists("demo_done.flag"):
        st.session_state.end = True
        os.remove("demo_done.flag")
        if st.session_state.audio_processor and not st.session_state.audio_processor.stopped:
            st.session_state.audio_processor.stop()
        st.rerun()

    if st.session_state.end:
        st.success("Demo ended. Loading final data...")
    else:
        audio_processor = st.session_state.audio_processor
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDRECV,
            audio_frame_callback=audio_processor.recv,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
        )
        with audio_processor.lock:
            if audio_processor.transcription:
                st.markdown(f"**Transcription:** {audio_processor.transcription}")
                audio_processor.transcription = None

    # --- HEADER ---
    log, _, tit_col_log, _ = st.columns([0.5, 0.2, 4, 2])
    with log:
        st.image("logo.png", use_container_width=True)
    with tit_col_log:
        st.markdown("<br><span class='Title'>Agatha control dashboard</span>", unsafe_allow_html=True)

    # DB & Data loaders
    @st.cache_resource
    def get_db_cached():
        return next(get_db())
    db = get_db_cached()

    # Toggle helpers
    def toggle_block_card(card_number):
        try:
            if is_card_blocked(db, card_number):
                remove_from_blocked(db, card_number)
            else:
                add_to_blocked(db, card_number)
        except Exception:
            db.rollback()
            raise

    def toggle_whitelist_card(card_number):
        try:
            if is_card_whitelisted(db, card_number):
                remove_from_whitelist(db, card_number)
            else:
                add_to_whitelist(db, card_number)
        except Exception:
            db.rollback()
            raise

    def toggle_password_reset(card_number):
        try:
            if has_password_reset(db, card_number):
                remove_password_reset(db, card_number)
            else:
                add_password_reset(db, card_number, reason="manual analyst action")
        except Exception:
            db.rollback()
            raise

    @st.cache_data(ttl=5)
    def load_alerts():
        return get_alerts(db, limit=10)

    @st.cache_data(ttl=5)
    def load_transactions(card_number):
        return db.query(Transaction).filter(
            Transaction.card_number == card_number
        ).order_by(Transaction.id.desc()).limit(10).all()

    alerts = []
    alert_index = st.session_state.alert_index
    if st.session_state.end:
        alerts = load_alerts()

    def load_alert(index):
        if index < 0 or index >= len(alerts):
            return None, None, None, None, None
        alert = alerts[index]
        tx = db.query(Transaction).filter(Transaction.id == alert.transaction_id).first()
        if not tx:
            return alert, None, None, None, None
        card_number = tx.card_number
        customer_name = f"{tx.customer_first_name} {tx.customer_last_name}" if tx.customer_first_name and tx.customer_last_name else "Unknown"
        notes = alert.analyst_notes or ""
        tx_list = load_transactions(card_number)
        return alert, card_number, customer_name, notes, tx_list

    alert, card_number, customer_name, notes, tx_list = load_alert(alert_index) if st.session_state.end else (None, None, None, None, None)

    if st.session_state.end:
        st.session_state.analyst_notes = notes or ""

    if not st.session_state.end:
        st.info("Demo running... Please wait until it finishes to see results.")
    else:
        if alert is None:
            st.markdown(page_style + "<span class='Title'>No alerts to display.</span><br>", unsafe_allow_html=True)
        else:
            whitelisted = is_card_whitelisted(db, card_number) is not None
            blocked = is_card_blocked(db, card_number) is not None
            reset = has_password_reset(db, card_number) is not None
            status_parts = []
            if whitelisted: status_parts.append("✅ Whitelisted")
            if blocked: status_parts.append("⛔ Blocked")
            if reset: status_parts.append("🔑 Password Reset")

            col1, _, _ = st.columns([2,1,1])
            with col1:
                st.markdown(
                    page_style +
                    f"<span class='subTitle'>🚨 Alert ID: {alert.id}</span><br>"
                    f"<span class='identificativoCost'>Customer:</span> <span class='valoreCost'>{customer_name}</span>"
                    f"<span class='identificativoCost'> | Transaction ID: </span><span class='valoreCost'>{alert.transaction_id}</span>",
                    unsafe_allow_html=True
                )

            col_conv, col_annotation = st.columns([1.5,1])
            with col_conv:
                with st.container(border=True, height=250):
                    st.markdown(tab_style + "<span class='subTitle'> 💬 Conversation Transcript</span>", unsafe_allow_html=True)
                    st.session_state.chat_messages = get_conversation(db, alert.id) or []
                    chat_html = ""
                    for msg in st.session_state.chat_messages:
                        role_class = "assistant" if msg.role == "assistant" else "user"
                        icon = "🤖" if msg.role == "assistant" else "👤"
                        content = msg.content.replace("\n", "<br>")
                        chat_html += f'<span class="chat-row {role_class}"><span class="icon">{icon}</span><span class="chat-bubble"><span class="message-text">{content}</span></span></span>'
                    st.markdown(chat_style + chat_html, unsafe_allow_html=True)

            with col_annotation:
                st.markdown("<span class='subTitle'>✍️ Analyst Notes & Actions</span>", unsafe_allow_html=True)
                with st.container(border=True, height=130):
                    text = st.session_state.analyst_notes.replace("$", "\$").replace("\n", "<br>")
                    st.markdown(f"<span class='notes'>{text}</span>", unsafe_allow_html=True)
                st.markdown(
                    f"<span class='identificativoCard'>Card:</span> <span class='valoreCard'>{card_number}</span><br>"
                    f"<span class='identificativoCard'>Action: </span><span class='valoreCard'>{' | '.join(status_parts) if status_parts else 'No special status'}</span>",
                    unsafe_allow_html=True
                )

            _, b1, b2, b3, _ = st.columns([1,1,1,1,1])
            st.markdown(bottom_style, unsafe_allow_html=True)
            if b1.button("Whitelist Card", icon="✅"):
                toggle_whitelist_card(card_number)
                st.rerun()
            if b2.button("Block Card", icon="⛔"):
                toggle_block_card(card_number)
                st.rerun()
            if b3.button("Reset Password", icon="🔑"):
                toggle_password_reset(card_number)
                st.rerun()

            with st.container(border=True, height=200):
                df = pd.DataFrame([{
                    "id": tx.id,
                    "Timestamp": tx.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    "Amount": tx.amount,
                    "Merchant": tx.merchant_name,
                    "Status": tx.status,
                    "Fraud Score": f"{tx.fraud_score:.2f}" if tx.fraud_score else "N/A",
                    "Fraudulent": tx.is_fraud,
                    "Alerted": "⚠️" if tx.id == alert.transaction_id else ""
                } for tx in tx_list])

                js_code = JsCode(f"""
                function(params) {{
                  if (params.data.id === {alert.transaction_id}) {{
                    return {{'fontWeight': 'bold', 'backgroundColor': '#ffff99'}};
                  }}
                  if (params.data.Fraudulent === true) {{
                    return {{'backgroundColor': '#ffcccc'}};
                  }}
                  return null;
                }}
                """)

                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_column("Fraudulent", editable=True, cellEditor='agCheckboxCellEditor')
                gb.configure_grid_options(getRowStyle=js_code)
                grid_options = gb.build()

                grid_response = AgGrid(df,
                                       gridOptions=grid_options,
                                       update_mode=GridUpdateMode.MODEL_CHANGED,
                                       allow_unsafe_jscode=True,
                                       fit_columns_on_grid_load=True,
                                       theme=StAggridTheme(base="alpine"),
                                       use_container_width=True)
                edited_df = grid_response['data']

            with st.container(border=True, height=66):
                _, c1, c2, c3, _ = st.columns([2,1,1,1,2])
                if c1.button("Previous Alert", icon="⬅️"):
                    st.session_state.alert_index = max(0, st.session_state.alert_index - 1)
                    st.rerun()
                if c2.button("Save Changes", icon="💾"):
                    update_alert(db, alert.id, analyst_notes=st.session_state.analyst_notes)
                    for _, row in edited_df.iterrows():
                        update_transaction(db, row['id'], is_fraud=row['Fraudulent'])
                    st.success("Changes saved.")
                    st.rerun()
                if c3.button("Next Alert", icon="➡️"):
                    st.session_state.alert_index = min(len(alerts)-1, st.session_state.alert_index + 1)
                    st.rerun()

else:

    st.info("Please enter your name and surname above and click 'Start Demo' to begin.")

