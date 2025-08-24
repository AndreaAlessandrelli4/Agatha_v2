import asyncio
import logging
import pathlib
from nltk.tokenize import sent_tokenize
from openai import AsyncOpenAI
from fraud_ai.config import OPENAI_API_KEY
from fraud_ai.prompt_builder import build_system_prompt
from fraud_ai.llm_agent import (
    llm_user_verification,
    llm_classify_user_reply,
    llm_classify_investigation_reply,
    llm_classify_help_reply,
    finalize_call_summary
)
from fraud_ai.conversation import add_message
from fraud_ai.STT import listen_and_transcribe
from fraud_ai.data import update_transaction
from fraud_ai.voice import speak_stream_text
from fraud_ai.voice_2 import tts_worker

# === SUPPRESS SQLALCHEMY LOGGING ===
for noisy in ("sqlalchemy", "sqlalchemy.engine", "sqlalchemy.pool", "sqlalchemy.dialects"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# === Terminal colors ===
RESET = "\033[0m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_YELLOW = "\033[93m"

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def stream_llm_with_tts(step_prompt, history, system_prompt, tts_backend="openai"):
    full_text = ""
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": step_prompt}]
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
                print(f"{BRIGHT_CYAN}{delta.content}{RESET}", end="", flush=True)
                full_text += delta.content
        print()
    elif tts_backend == "openai":
        buffer = ""
        async for event in stream:
            delta = event.choices[0].delta
            if delta and delta.content:
                print(f"{BRIGHT_CYAN}{delta.content}{RESET}", end="", flush=True)
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
                print(f"{BRIGHT_CYAN}{delta.content}{RESET}", end="", flush=True)
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


async def ask_and_classify(db, alert_id, history, step_prompt, system_prompt, classifier_func,
                           tts_backend="text", stt_enabled=False, stt_provider="openai",
                           retry_limit=0):
    attempts = 0
    while True:
        if step_prompt.strip():
            assistant_text = await stream_llm_with_tts(step_prompt, history, system_prompt, tts_backend)
            add_message(db, alert_id, "assistant", assistant_text)
            history.append({"role": "assistant", "content": assistant_text})

        if stt_enabled:
            user_text = listen_and_transcribe(
                stt_enabled=stt_enabled, stt_provider=stt_provider
            ).strip()
            print(f"{BRIGHT_YELLOW}{user_text}{RESET}")
        else:
            user_text = input(f"{BRIGHT_YELLOW}Customer says:{RESET} ").strip()

        if not user_text:
            classification = "REPEAT"
        else:
            add_message(db, alert_id, "user", user_text)
            history.append({"role": "user", "content": user_text})
            print(f"\n[DEBUG] Using classifier: {classifier_func.__name__}")
            classification = classifier_func(user_text, history, system_prompt)
            print(f"[DEBUG] Classification result: {classification}")

        classification = classification.upper() if classification else "REPEAT"

        if retry_limit and classification in ("REPEAT", "OFFTOPIC", "CLARIFY"):
            attempts += 1
            if attempts >= retry_limit:
                return user_text, classification, True
            continue
        return user_text, classification, False


async def handle_end_classification(classification, history, system_prompt, tts_backend, db=None, alert_id=None):
    """
    Always generate and add a final goodbye or callback acknowledgement to history + DB.
    """
    text = ""
    if classification == "END":
        text = await stream_llm_with_tts(
            "End the call politely, confirm they'll get updates in the banking app, and remind about scam safety.",
            history, system_prompt, tts_backend
        )
    elif classification == "CALL_BACK_LATER":
        text = await stream_llm_with_tts(
            "Acknowledge their request and confirm you'll call them back later. End call politely.",
            history, system_prompt, tts_backend
        )
    elif classification == "NO_CALL_BACK":
        text = await stream_llm_with_tts(
            "Acknowledge that they do not wish to be called back and end the call politely.",
            history, system_prompt, tts_backend
        )
    elif classification == "CANT_TALK":
        text = await stream_llm_with_tts(
            "Acknowledge they cannot talk right now, confirm you'll try again later, and end politely.",
            history, system_prompt, tts_backend
        )

    # ✅ Always store assistant goodbye message
    if text:
        if db and alert_id:
            add_message(db, alert_id, "assistant", text)
        history.append({"role": "assistant", "content": text})


async def full_fraud_flow(db, alert, alerted_tx, recent_txs,
                          tts_backend="text", stt_enabled=False, stt_provider="openai"):
    history = []

    # === GREETING ===
    greet_system_prompt = build_system_prompt(alerted_tx, recent_txs, greeting_mode=True)
    _, classification, _ = await ask_and_classify(
        db, alert.id, history,
        f"Politely greet the customer, present yourself, and confirm if you are speaking to {alerted_tx.customer_first_name} {alerted_tx.customer_last_name}, the cardholder.",
        greet_system_prompt,
        llm_user_verification,
        tts_backend, stt_enabled, stt_provider,
        retry_limit=2
    )
    classification = classification.upper() if classification else "REPEAT"

    # === Handle NO / ENDINGS in greeting ===
    if classification == "NO":
        text = await stream_llm_with_tts(
            "Explain politely that you can only speak with the cardholder. "
            "Say you’ll try to call back later, and end the call politely. (don't refer to the person by cardholder name since he's not the cardholder)",
            history, greet_system_prompt, tts_backend
        )
        if db and alert.id:
            add_message(db, alert.id, "assistant", text)
        history.append({"role": "assistant", "content": text})

        final_result = finalize_call_summary(db, alert, alerted_tx, history)
        print("\n===== FINAL CALL SUMMARY =====")
        print(final_result["summary"])
        print("Actions decided:", final_result["actions"])
        print("================================\n")
        return False

    if classification in ("END", "CALL_BACK_LATER", "NO_CALL_BACK", "CANT_TALK"):
        await handle_end_classification(classification, history, greet_system_prompt, tts_backend, db, alert.id)
        final_result = finalize_call_summary(db, alert, alerted_tx, history)
        print("\n===== FINAL CALL SUMMARY =====")
        print(final_result["summary"])
        print("Actions decided:", final_result["actions"])
        print("================================\n")
        return False

    if classification != "YES":
        # fallback if identity could not be established
        text = await stream_llm_with_tts(
            "Identity could not be verified so the call will end. "
            "Tell you can talk only with the card holder (remember to not call the person on the phone by cardholder name since he's not the cardholder)"
            "End politely.",
            history, greet_system_prompt, tts_backend
        )
        if db and alert.id:
            add_message(db, alert.id, "assistant", text)
        history.append({"role": "assistant", "content": text})

        final_result = finalize_call_summary(db, alert, alerted_tx, history)
        print("\n===== FINAL CALL SUMMARY =====")
        print(final_result["summary"])
        print("Actions decided:", final_result["actions"])
        print("================================\n")
        return False

    # === TRANSACTION VERIFICATION ===
    tx_system_prompt = build_system_prompt(alerted_tx, recent_txs, greeting_mode=False)
    _, tx_result, _ = await ask_and_classify(
        db, alert.id, history,
        f"Tell the customer you’re calling because the fraud prevention system declined a transaction considered at risk. "
        f"Ask them to confirm if they authorised ${alerted_tx.amount:.2f} at {alerted_tx.merchant_name} on {alerted_tx.timestamp.strftime('%Y-%m-%d %H:%M')}.",
        tx_system_prompt,
        llm_classify_user_reply,
        tts_backend, stt_enabled, stt_provider,
        retry_limit=3
    )
    tx_result = tx_result.upper() if tx_result else "REPEAT"
    if tx_result in ("END", "CALL_BACK_LATER", "NO_CALL_BACK", "CANT_TALK"):
        await handle_end_classification(tx_result, history, tx_system_prompt, tts_backend, db, alert.id)
        final_result = finalize_call_summary(db, alert, alerted_tx, history)
        print("\n===== FINAL CALL SUMMARY =====")
        print(final_result["summary"])
        print("Actions decided:", final_result["actions"])
        print("================================\n")
        return False
    print(f"[DEBUG] Verification classification: {tx_result}")

    if tx_result == "OK":
        await stream_llm_with_tts(
            "Answer the customer's clarification directly and completely. DO NOT ask if they authorised the transaction here.",
            history, tx_system_prompt, tts_backend
        )
        _, tx_result, _ = await ask_and_classify(
            db, alert.id, history,
            "Please now confirm if you authorised that transaction.",
            tx_system_prompt,
            llm_classify_user_reply,
            tts_backend, stt_enabled, stt_provider,
            retry_limit=2
        )
        tx_result = tx_result.upper() if tx_result else "REPEAT"
        if tx_result in ("END", "CALL_BACK_LATER", "NO_CALL_BACK", "CANT_TALK"):
            await handle_end_classification(tx_result, history, tx_system_prompt, tts_backend, db, alert.id)
            final_result = finalize_call_summary(db, alert, alerted_tx, history)
            print("\n===== FINAL CALL SUMMARY =====")
            print(final_result["summary"])
            print("Actions decided:", final_result["actions"])
            print("================================\n")
            return False
        print(f"[DEBUG] Verification-after-OK classification: {tx_result}")

    # === FRAUD path ===
    if tx_result == "FRAUD":
        update_transaction(db, alerted_tx.id, is_fraud=True)

        # Investigation loop
        while True:
            _, inv_class, _ = await ask_and_classify(
                db, alert.id, history,
                """Thank the customer for confirming the transaction was fraudulent. 
                Reassure them protective steps will be taken: block the card, monitor suspicious activity, and perform an investigation. 
                Ask if they've noticed any suspicious emails, SMS, or calls from people pretending to be bank staff, 
                and whether they have entered their card data on unfamiliar websites""",
                tx_system_prompt,
                llm_classify_investigation_reply,
                tts_backend, stt_enabled, stt_provider
            )
            inv_class = inv_class.upper() if inv_class else "REPEAT"
            if inv_class in ("END", "CALL_BACK_LATER", "NO_CALL_BACK", "CANT_TALK"):
                await handle_end_classification(inv_class, history, tx_system_prompt, tts_backend, db, alert.id)
                break
            if inv_class == "INFO_COMPLETE":
                break
            elif inv_class in ("REPEAT", "OFFTOPIC", "INFO_INCOMPLETE"):
                continue
            else:
                break

        # Secondary tx check
        for tx in recent_txs:
            if tx.id == alerted_tx.id:
                continue
            _, other_result, _ = await ask_and_classify(
                db, alert.id, history,
                f"Ask if they authorised ${tx.amount:.2f} at {tx.merchant_name} on {tx.timestamp.strftime('%Y-%m-%d %H:%M')}.",
                tx_system_prompt,
                llm_classify_user_reply,
                tts_backend, stt_enabled, stt_provider,
                retry_limit=1
            )
            other_result = other_result.upper() if other_result else "REPEAT"
            if other_result in ("END", "CALL_BACK_LATER", "NO_CALL_BACK", "CANT_TALK"):
                await handle_end_classification(other_result, history, tx_system_prompt, tts_backend, db, alert.id)
                break
            if other_result == "FRAUD":
                update_transaction(db, tx.id, is_fraud=True)
            elif other_result == "NOT FRAUD":
                update_transaction(db, tx.id, is_fraud=False)

    elif tx_result == "NOT FRAUD":
        update_transaction(db, alerted_tx.id, is_fraud=False)

    # === HELP-OFFER LOOP ===
    help_attempts = 0
    while True:
        prompt_to_use = (
            "Summarise outcome and ask if they need any other assistance."
            if help_attempts == 0
            else "Do they need any more assistance before ending?"
        )
        _, help_class, _ = await ask_and_classify(
            db, alert.id, history,
            prompt_to_use,
            tx_system_prompt,
            llm_classify_help_reply,
            tts_backend, stt_enabled, stt_provider
        )
        help_class = help_class.upper() if help_class else "REPEAT"
        print(f"[DEBUG] Help-offer classification: {help_class}")

        if help_class in ("END", "CALL_BACK_LATER", "NO_CALL_BACK", "CANT_TALK"):
            await handle_end_classification(help_class, history, tx_system_prompt, tts_backend, db, alert.id)
            break

        if help_class == "YES":
            await stream_llm_with_tts(
                "Handle their request in detail. Ask if they need any other assistance.",
                history, tx_system_prompt, tts_backend
            )

            if stt_enabled:
                user_text = listen_and_transcribe(
                    stt_enabled=stt_enabled, stt_provider=stt_provider
                ).strip()
                print(f"{BRIGHT_YELLOW}{user_text}{RESET}")
            else:
                user_text = input(f"{BRIGHT_YELLOW}Customer says:{RESET} ").strip()

            add_message(db, alert.id, "user", user_text)
            history.append({"role": "user", "content": user_text})

            follow_up = llm_classify_help_reply(user_text, history, tx_system_prompt)
            follow_up = follow_up.upper() if follow_up else "REPEAT"
            print(f"[DEBUG] Follow-up after YES classification: {follow_up}")

            if follow_up in ("END", "CALL_BACK_LATER", "NO_CALL_BACK", "CANT_TALK"):
                await handle_end_classification(follow_up, history, tx_system_prompt, tts_backend, db, alert.id)
                break
            if follow_up == "NO":
                await handle_end_classification("END", history, tx_system_prompt, tts_backend, db, alert.id)
                break
            if follow_up in ("REPEAT", "OFFTOPIC", "CLARIFY"):
                help_attempts += 1
                if help_attempts >= 3:
                    await handle_end_classification("END", history, tx_system_prompt, tts_backend, db, alert.id)
                    break
                continue

            help_attempts = 1
            continue

        if help_class == "NO":
            await handle_end_classification("END", history, tx_system_prompt, tts_backend, db, alert.id)
            break

        if help_class in ("REPEAT", "OFFTOPIC", "CLARIFY"):
            help_attempts += 1
            if help_attempts >= 3:
                await handle_end_classification("END", history, tx_system_prompt, tts_backend, db, alert.id)
                break
            continue

        help_attempts = 1

    final_result = finalize_call_summary(db, alert, alerted_tx, history)
    print("\n===== FINAL CALL SUMMARY =====")
    print(final_result.get("summary", "No summary available."))

    actions = final_result.get("actions") or []
    print("Actions decided:", actions)
    print("================================\n")

    # Signal demo end by creating the flag file
    import pathlib
    pathlib.Path("demo_done.flag").write_text("done")