import asyncio
from openai import AsyncOpenAI
from nltk.tokenize import sent_tokenize
from fraud_ai.config import OPENAI_API_KEY
from fraud_ai.conversation import add_message
from fraud_ai.prompt_builder import build_system_prompt
from fraud_ai.voice import speak_stream_text
from fraud_ai.voice_2 import tts_worker
from fraud_ai.conversation_config import conversation_handlers, verification_handlers, CONVERSATION_FLOW
from fraud_ai.STT import listen_and_transcribe

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def stream_llm_with_tts(step_prompt, history, system_prompt, tts_backend="openai"):
    """Streams GPT output with persistent system prompt."""
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
                print(delta.content, end="", flush=True)
                full_text += delta.content
        print()

    elif tts_backend == "openai":
        buffer = ""
        async for event in stream:
            delta = event.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)
                buffer += delta.content
                full_text += delta.content
        print()
        await speak_stream_text(buffer.strip())

    elif tts_backend == "elevenlabs":
        tts_queue = asyncio.Queue()
        tts_task = asyncio.create_task(tts_worker(tts_queue))
        buffer = ""
        async for event in stream:
            delta = event.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)
                buffer += delta.content
                full_text += delta.content
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


async def send_and_log_stream(
    db, alert_id, history, step_prompt, tts_backend, system_prompt
):
    text = await stream_llm_with_tts(
        step_prompt,
        history,
        system_prompt,
        tts_backend
    )
    add_message(db, alert_id, "assistant", text)
    history.append({"role": "assistant", "content": text})


def wait_for_reply_sync(
    db, alert_id, history, system_prompt,
    llm_classifier, handlers,
    stt_enabled, stt_provider,
    max_attempts=3
):
    attempts = 0
    while True:
        customer_input = listen_and_transcribe(
            stt_enabled=stt_enabled, stt_provider=stt_provider
        ).strip()
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


async def conversation_manager(
    db,
    alert_id,
    history,
    alerted_tx,
    recent_txs,
    tts_backend="text",
    stt_enabled=True,
    stt_provider="openai",
    greeting_mode=False
):
    """
    In greeting_mode:
      - Force llm_user_verification classifier
      - YES -> return True (proceed to transactions)
      - NO/REPEAT/OFFTOPIC/CLARIFY -> retry max_attempts, else goodbye and return False.
    In normal mode: Run CONVERSATION_FLOW as defined.
    """
    from fraud_ai.llm_agent import llm_user_verification

    system_prompt = build_system_prompt(alerted_tx, recent_txs, greeting_mode=greeting_mode)
    handlers_map = {
        "conversation_handlers": conversation_handlers,
        "verification_handlers": verification_handlers
    }
    current_state = "GREETING"

    attempts = 0
    max_attempts = 2

    while current_state:
        step = next(s for s in CONVERSATION_FLOW if s["name"] == current_state)

        if step["prompt"]:
            if step["name"] == "GREETING" and alerted_tx:
                first = alerted_tx.customer_first_name
                last = alerted_tx.customer_last_name
                step_prompt = f"Hello, I'm Agata, the AI Fraud Analyst from SAS Bank. Am I speaking to {first} {last}?"
            else:
                step_prompt = step["prompt"]

            await send_and_log_stream(
                db, alert_id, history,
                step_prompt,
                tts_backend,
                system_prompt
            )

        if not step["classifier"]:
            break

        # If greeting mode, force verification classifier
        classifier_to_use = llm_user_verification if greeting_mode else step["classifier"]

        _, classification, end = wait_for_reply_sync(
            db,
            alert_id,
            history,
            system_prompt,
            classifier_to_use,
            handlers_map[step["handlers"]],
            stt_enabled,
            stt_provider
        )

        if greeting_mode:
            classification = classification.upper() if classification else "REPEAT"
            print(f"[Verification classifier result]: {classification}")

            if classification == "YES":
                return True

            if classification == "NO":
                # Fixed hard-coded farewell (no LLM freelancing)
                await send_and_log_stream(
                    db, alert_id, history,
                    "I'm sorry, but I must speak directly with the cardholder. "
                    "I will call back later when they are available. Goodbye.",
                    tts_backend,
                    system_prompt
                )
                return False

            if classification in ("REPEAT", "OFFTOPIC", "CLARIFY"):
                attempts += 1
                if attempts >= max_attempts:
                    await send_and_log_stream(
                        db, alert_id, history,
                        "It seems we cannot confirm your identity at this time. "
                        "Please have the cardholder contact us or we will try again later. Goodbye.",
                        tts_backend,
                        system_prompt
                    )
                    return False
                continue

            # Any other unexpected classification
            await send_and_log_stream(
                db, alert_id, history,
                "I'm sorry, but I must speak directly with the cardholder. "
                "I will call back later when they are available. Goodbye.",
                tts_backend,
                system_prompt
            )
            return False

        if end:
            break

        current_state = step["next"]

    return True