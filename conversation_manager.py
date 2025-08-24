import os
import asyncio
from datetime import timedelta
from sqlalchemy.orm import Session

from fraud_ai.llm_agent import llm_classify_user_reply, llm_user_verification
from fraud_ai.response_handlers import (
    handle_repeat, handle_offtopic, handle_end,
    handle_cant_talk, handle_call_back_later, handle_no_call_back
)
from openai import AsyncOpenAI
from fraud_ai.config import OPENAI_API_KEY
from fraud_ai.voice import speak_stream_text
from fraud_ai.voice_2 import tts_worker
from nltk.tokenize import sent_tokenize
from fraud_ai.prompt_builder import build_system_prompt
from fraud_ai.conversation import add_message
from fraud_ai.data import init_db, get_db, create_transaction, update_transaction, Transaction
from fraud_ai.alerts import create_alert, get_alerts

# ==== Runtime I/O config ====
DEFAULT_TTS_BACKEND = "elevenlabs"  # Options: "elevenlabs", "openai", "text"
DEFAULT_STT_BACKEND = "text"        # Reserved for future speech-to-text config
DEFAULT_INPUT_MODE = "text"         # "text" or "microphone" in the future

# Async OpenAI client for streaming
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# === Handler maps ===
conversation_handlers = {
    "REPEAT": handle_repeat,
    "OFFTOPIC": handle_offtopic,
    "END": handle_end,
    "CANT_TALK": handle_cant_talk,
    "CALL_BACK_LATER": handle_call_back_later,
    "NO_CALL_BACK": handle_no_call_back
}
verification_handlers = conversation_handlers.copy()

# === Conversation state flow ===
CONVERSATION_FLOW = [
    {
        "name": "GREETING",
        "prompt": "Politely greet the customer and introduce yourself as Agata the AI Fraud Analyst of SAS Bank. Ask if you're talking with the cardholder.",
        "classifier": llm_user_verification,
        "handlers": "conversation_handlers",
        "next": "MAIN_CONVO"
    },
    {
        "name": "VERIFICATION",
        "prompt": "Politely ask if you are speaking to the cardholder.",
        "classifier": llm_user_verification,
        "handlers": "verification_handlers",
        "next": "MAIN_CONVO"
    },
    {
        "name": "MAIN_CONVO",
        "prompt": "Proceed with explaining the suspicious activity and ask relevant questions or furnish relevant explanations.",
        "classifier": llm_classify_user_reply,
        "handlers": "conversation_handlers",
        "next": "END"
    },
    {
        "name": "END",
        "prompt": "Politely thank the customer and say goodbye.",
        "classifier": None,
        "handlers": None,
        "next": None
    }
]

# === Streaming and reply helpers ===
async def stream_llm_with_tts(step_prompt: str, history: list, tts_backend: str, system_prompt: str):
    full_text = ""
    messages = (
        [{"role": "system", "content": system_prompt}] +
        history +
        [{"role": "user", "content": step_prompt}]
    )
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    if tts_backend == "text":
        async for event in stream:
            delta = event.choices[0].delta
            if delta and delta.content:
                token = delta.content
                print(token, end="", flush=True)
                full_text += token
        print()
    elif tts_backend == "openai":
        buffer = ""
        async for event in stream:
            delta = event.choices[0].delta
            if delta and delta.content:
                token = delta.content
                print(token, end="", flush=True)
                buffer += token
                full_text += token
        print()
        await speak_stream_text(buffer.strip())
    elif tts_backend == "elevenlabs":
        tts_queue = asyncio.Queue()
        tts_task = asyncio.create_task(tts_worker(tts_queue))
        buffer = ""
        async for event in stream:
            delta = event.choices[0].delta
            if delta and delta.content:
                token = delta.content
                print(token, end="", flush=True)
                buffer += token
                full_text += token
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
    return full_text.strip()

async def send_and_log_stream(db, alert_id, history, step_prompt, tts_backend, system_prompt):
    text = await stream_llm_with_tts(step_prompt, history, tts_backend, system_prompt)
    add_message(db, alert_id, "assistant", text)
    history.append({"role": "assistant", "content": text})

def wait_for_reply_sync(db, alert_id, history, system_prompt, llm_classifier, handlers, max_attempts=3):
    attempts = 0
    while True:
        customer_input = input("Customer: ").strip()
        if not customer_input:
            attempts += 1
            print("Please enter a reply.")
            if attempts >= max_attempts:
                return None, "NO_ANSWER", True
            continue
        add_message(db, alert_id, "user", customer_input)
        history.append({"role": "user", "content": customer_input})
        classification = llm_classifier(customer_input, history, system_prompt)
        print(f"[LLM classification]: {classification}")
        if classification in handlers:
            return handlers[classification](db, alert_id, history, system_prompt)
        return customer_input, classification, False

# === Conversation manager ===
async def conversation_manager(db, alert_id, history, alerted_tx, recent_txs, tts_backend=DEFAULT_TTS_BACKEND):
    system_prompt = build_system_prompt(alerted_tx, recent_txs)
    handlers_map = {
        "conversation_handlers": conversation_handlers,
        "verification_handlers": verification_handlers
    }
    current_state = "GREETING"
    while current_state:
        step = next(s for s in CONVERSATION_FLOW if s["name"] == current_state)
        if step["prompt"]:
            await send_and_log_stream(
                db, alert_id, history,
                step["prompt"], tts_backend, system_prompt
            )
        if not step["classifier"]:
            break
        _, classification, end = wait_for_reply_sync(
            db, alert_id, history, system_prompt,
            step["classifier"],
            handlers_map[step["handlers"]]
        )
        if end:
            break
        if current_state == "MAIN_CONVO":
            if classification == "CONFIRMED_FRAUD":
                if alerted_tx:
                    update_transaction(db, alerted_tx.id, is_fraud=True)
            elif classification == "NOT_FRAUD":
                if alerted_tx:
                    update_transaction(db, alerted_tx.id, is_fraud=False)
        current_state = step["next"]

# === High-level orchestrator steps ===
async def verify_single_transaction(db, alert_id, tx, recent_txs):
    history = []
    await conversation_manager(
        db=db,
        alert_id=alert_id,
        history=history,
        alerted_tx=tx,
        recent_txs=recent_txs,
        tts_backend=DEFAULT_TTS_BACKEND
    )
    db.refresh(tx)
    return tx.is_fraud

async def apology_step(db, alert_id, alerted_tx, recent_txs):
    history = []
    system_prompt = build_system_prompt(alerted_tx, recent_txs)
    await send_and_log_stream(
        db, alert_id, history,
        "Apologize politely, explain the transaction looks okay, and advise the customer to retry later if the issue persists.",
        tts_backend=DEFAULT_TTS_BACKEND,
        system_prompt=system_prompt
    )

async def ask_if_needs_help(db, alert_id, alerted_tx, recent_txs):
    history = []
    system_prompt = build_system_prompt(alerted_tx, recent_txs)
    await send_and_log_stream(
        db, alert_id, history,
        "Before we finish, ask politely if the customer needs anything else.",
        tts_backend=DEFAULT_TTS_BACKEND,
        system_prompt=system_prompt
    )
    reply = input("Customer: ")
    classification = llm_classify_user_reply(reply, history, system_prompt)
    return classification.lower() in ["yes", "affirmative", "y"]

async def furnish_help(db, alert_id, alerted_tx, recent_txs):
    history = []
    system_prompt = build_system_prompt(alerted_tx, recent_txs)
    await send_and_log_stream(
        db, alert_id, history,
        "Provide further explanation or assist the customer as requested.",
        tts_backend=DEFAULT_TTS_BACKEND,
        system_prompt=system_prompt
    )

async def polite_goodbye(db, alert_id, alerted_tx, recent_txs):
    history = []
    system_prompt = build_system_prompt(alerted_tx, recent_txs)
    await send_and_log_stream(
        db, alert_id, history,
        "Politely thank the customer for their time and say goodbye.",
        tts_backend=DEFAULT_TTS_BACKEND,
        system_prompt=system_prompt
    )

# === Master fraud-check flow ===
async def full_fraud_flow(db, alert, alerted_tx, recent_txs):
    fraud_confirmed = await verify_single_transaction(db, alert.id, alerted_tx, recent_txs)
    if fraud_confirmed:
        non_alerted_recent = [tx for tx in recent_txs if tx.id != alerted_tx.id]
        for tx in non_alerted_recent:
            await verify_single_transaction(db, None, tx, recent_txs)
    else:
        await apology_step(db, alert.id, alerted_tx, recent_txs)
    if await ask_if_needs_help(db, alert.id, alerted_tx, recent_txs):
        await furnish_help(db, alert.id, alerted_tx, recent_txs)
    else:
        await polite_goodbye(db, alert.id, alerted_tx, recent_txs)

# === Main script ===
if __name__ == "__main__":
    os.remove("fraud_ai.db") if os.path.exists("fraud_ai.db") else None
    init_db()
    print("Database initialized!")
    db = next(get_db())

    card_number = "9999888877776666"
    stores = ["Amazon", "Google", "Netflix"]
    customer = {"first_name": "John", "last_name": "Doe"}

    # Create demo transactions
    for i in range(3):
        status = "approved" if i < 2 else "declined"
        tx = create_transaction(
            db,
            card_number=card_number,
            amount=100 + i * 50,
            fraud_score=0.2 * i,
            is_fraud=False,
            status=status,
            merchant_id=f"M{i+100}",
            merchant_name=stores[i],
            mcc="5999",
            country="US",
            id=i,
            customer_first_name=customer["first_name"],
            customer_last_name=customer["last_name"]
        )
        print(f"Inserted transaction {tx.id} ({status})")

    # Create alert on last transaction
    last_tx = db.query(Transaction).filter(Transaction.id == 2).first()
    if last_tx:
        alert = create_alert(
            db,
            transaction_id=last_tx.id,
            status="open",
            analyst_notes="Auto-generated alert for declined transaction"
        )
        print(f"Created alert {alert.id} for tx {last_tx.id}")

    alerts = get_alerts(db)
    for a in alerts:
        print(f"Alert {a.id} for transaction {a.transaction_id}")

    def get_transactions_last_24h(db: Session, card_number: str, alerted_tx_time, window_hours=24):
        start_time = alerted_tx_time - timedelta(hours=window_hours)
        end_time = alerted_tx_time + timedelta(hours=window_hours)
        return db.query(Transaction).filter(
            Transaction.card_number == card_number,
            Transaction.timestamp >= start_time,
            Transaction.timestamp <= end_time
        ).order_by(Transaction.timestamp.asc()).all()

    alerted_tx = last_tx
    recent_txs = get_transactions_last_24h(db, card_number, alerted_tx.timestamp)

    asyncio.run(full_fraud_flow(db, alert, alerted_tx, recent_txs))