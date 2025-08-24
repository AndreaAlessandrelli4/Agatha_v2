from fraud_ai.data import init_db
import os 
os.remove("fraud_ai.db")

try:
    if __name__ == "__main__":
        init_db()
        print("Database initialized!")
except:
    pass


from fraud_ai.data import get_db, create_transaction
from fraud_ai.alerts import create_alert, get_alerts
from fraud_ai.data import update_transaction

db = next(get_db())

card_number = "9999888877776666"  # Same card as your alert

stores = ["Amazon", "Google", "Netflix"]

# Insert multiple transactions for the same card and create alerts for them
for i in range(3):
    tx = create_transaction(
        db,
        card_number=card_number,
        amount=100 + i * 50,  # Vary amounts
        fraud_score=0.5 * i,  # Vary fraud scores
        is_fraud=False,
        status="completed" if i % 2 == 0 else "pending",
        merchant_id=f"M{i+100}",
        merchant_name=f"{stores[i]}",
        mcc="5999",
        country="US",
        id=i
    )
    print(f"Inserted transaction {tx.id} for card {card_number}")

    # Create alert if fraud_score > 0.3 (example threshold)
    if tx.fraud_score > 0.3:
        alert = create_alert(db, transaction_id=tx.id, status="open", analyst_notes="Auto-generated alert")
        print(f"Created alert {alert.id} for transaction {tx.id}")














# Verify alerts exist
alerts = get_alerts(db)
print("Current alerts:")
for a in alerts:
    print(f"Alert {a.id} for transaction {a.transaction_id}, status: {a.status}, notes: {a.analyst_notes}")

# demo_call.py
from datetime import timedelta
from fraud_ai.data import get_db, update_transaction
from fraud_ai.alerts import get_alert, update_alert
from fraud_ai.conversation import add_message, get_conversation
from fraud_ai.models import Transaction
from fraud_ai.llm_agent import chatgpt_response, llm_classify_user_reply
from fraud_ai.prompt_builder import build_system_prompt
from sqlalchemy.orm import Session
import time

def get_transactions_last_24h(db: Session, card_number: str, alerted_tx_time, window_hours=24):
    start_time = alerted_tx_time - timedelta(hours=window_hours)
    end_time = alerted_tx_time + timedelta(hours=window_hours)
    return db.query(Transaction).filter(
        Transaction.card_number == card_number,
        Transaction.timestamp >= start_time,
        Transaction.timestamp <= end_time
    ).order_by(Transaction.timestamp.asc()).all()


def wait_for_valid_reply(db: Session, alert_id: int, history: list, system_prompt: str, max_attempts: int = 3):
    """
    Wait for user input (simulated via input() here). Returns (text, classification, ended).
    classification can be: OK, FRAUD, REPEAT, OFFTOPIC, END, CANT_TALK, CALL_BACK_LATER, NO_CALL_BACK
    If ended == True -> terminate the call.
    """
    attempts = 0
    while True:
        customer_input = input("Customer: ").strip()
        if not customer_input:
            attempts += 1
            print("Please enter a reply.")
            if attempts >= max_attempts:
                print("[No valid reply received — giving up on this attempt]")
                return None, "NO_ANSWER", True
            continue

        # Log customer input
        add_message(db, alert_id, "user", customer_input)
        history.append({"role": "user", "content": customer_input})

        # Classify the reply
        classification = llm_classify_user_reply(customer_input, "en", history, system_prompt)
        print(f"[LLM classification]: {classification}")

        # Handling REPEAT
        if classification == "REPEAT":
            repeat_msg = chatgpt_response(history, "Please politely ask the customer to repeat their last response.")
            print(f"Agata: {repeat_msg}")
            add_message(db, alert_id, "assistant", repeat_msg)
            history.append({"role": "assistant", "content": repeat_msg})
            attempts += 1
            if attempts >= max_attempts:
                return None, "NO_ANSWER", True
            continue

        # Handling OFFTOPIC
        if classification == "OFFTOPIC":
            off_msg = chatgpt_response(history, "Please politely bring the customer back to verifying recent transactions.")
            print(f"Agata: {off_msg}")
            add_message(db, alert_id, "assistant", off_msg)
            history.append({"role": "assistant", "content": off_msg})
            attempts += 1
            if attempts >= max_attempts:
                return None, "NO_ANSWER", True
            continue

        # Handling END
        if classification == "END":
            farewell = chatgpt_response(history, "Please politely thank the customer and say goodbye.")
            print(f"Agata: {farewell}")
            add_message(db, alert_id, "assistant", farewell)
            history.append({"role": "assistant", "content": farewell})
            return None, "END", True

        # New path: Customer can't talk right now
        if classification == "CANT_TALK":
            follow_up_msg = chatgpt_response(history, "Politely ask the customer if you can call them back later.")
            print(f"Agata: {follow_up_msg}")
            add_message(db, alert_id, "assistant", follow_up_msg)
            history.append({"role": "assistant", "content": follow_up_msg})

            # Wait for their answer
            return wait_for_valid_reply(db, alert_id, history, system_prompt)

        if classification == "CALL_BACK_LATER":
            confirm_msg = chatgpt_response(history, "Politely thank the customer and confirm that you will call them back later. Say goodbye.")
            print(f"Agata: {confirm_msg}")
            add_message(db, alert_id, "assistant", confirm_msg)
            history.append({"role": "assistant", "content": confirm_msg})
            return None, "CALL_BACK_LATER", True

        if classification == "NO_CALL_BACK":
            fallback_msg = chatgpt_response(history, "Politely inform the customer you will send them an email with the details instead. Say goodbye.")
            print(f"Agata: {fallback_msg}")
            add_message(db, alert_id, "assistant", fallback_msg)
            history.append({"role": "assistant", "content": fallback_msg})
            # TODO: trigger email send here
            return None, "NO_CALL_BACK", True

        # Otherwise (OK, FRAUD, etc.)
        return customer_input, classification, False


def wrap_up_call(db: Session, alert_id: int, history: list, outcome: str):
    """
    outcome: 'FRAUD' or 'NOT_FRAUD' or other descriptive string.
    Ask LLM to generate a polite wrap-up (apology if needed, next steps, offer help, goodbye).
    """
    if outcome == "FRAUD":
        wrap_prompt = (
            "Please thank the customer, apologize for the inconvenience, explain that we will block the card "
            "and any further steps the bank will take (e.g., notifications in the app), and offer help for anything else. "
            "End with a polite goodbye."
        )
    else:
        wrap_prompt = (
            "Please thank the customer for confirming, apologize for the inconvenience of the verification, "
            "reassure them their account is secure, tell them they will receive a notification in the app, "
            "ask if they need anything else, and say goodbye."
        )

    closing_message = chatgpt_response(history, wrap_prompt)
    print(f"Agata: {closing_message}")
    add_message(db, alert_id, "assistant", closing_message)
    history.append({"role": "assistant", "content": closing_message})

    # Optionally add a final "Call Ended" assistant entry if you want to store an explicit flag
    add_message(db, alert_id, "assistant", "Call Ended")
    history.append({"role": "assistant", "content": "Call Ended"})


def simulate_full_call(db: Session, alert_id: int, system_prompt: str, alerted_tx: Transaction):
    print("Starting simulated call with Agata...\n")

    conversation_msgs = get_conversation(db, alert_id)
    history = [{"role": m.role, "content": m.content} for m in conversation_msgs]

    # Put system prompt as a system message for context (so chatgpt_response has it)
    if system_prompt:
        history.insert(0, {"role": "system", "content": system_prompt})

    if not any(m['role'] == 'assistant' for m in history):
        greeting = "Hello, this is Agata from your bank's fraud department. I’d like to verify a recent transaction with you."
        add_message(db, alert_id, "assistant", greeting)
        history.append({"role": "assistant", "content": greeting})
        print(f"Agata: {greeting}")

    # Ask to verify the alerted transaction
    prompt = f"Please verify this transaction: ${alerted_tx.amount:.2f} at {alerted_tx.merchant_name} on {alerted_tx.timestamp.strftime('%Y-%m-%d %H:%M:%S')}."
    add_message(db, alert_id, "assistant", prompt)
    history.append({"role": "assistant", "content": prompt})
    print(f"Agata: {prompt}")

    customer_input, classification, ended = wait_for_valid_reply(db, alert_id, history, system_prompt)
    if ended:
        if classification in ["NO_ANSWER", "CALL_BACK_LATER", "NO_CALL_BACK", "END"]:
            print(f"[Call ended with classification: {classification}]")
            return

    # Map OK -> NOT_FRAUD semantic
    if classification == "OK":
        classification = "NOT_FRAUD"

    if classification == "FRAUD":
        update_transaction(db, alerted_tx.id, is_fraud=True)
        print(f"Transaction {alerted_tx.id} marked as FRAUDULENT.")

        # Fetch other transactions within 24h
        other_txs = get_transactions_last_24h(db, alerted_tx.card_number, alerted_tx.timestamp)
        # Exclude alerted_tx itself
        transactions_to_verify = [tx for tx in other_txs if tx.id != alerted_tx.id]

        for tx in transactions_to_verify:
            prompt = f"Please verify this transaction: ${tx.amount:.2f} at {tx.merchant_name} on {tx.timestamp.strftime('%Y-%m-%d %H:%M:%S')}."
            add_message(db, alert_id, "assistant", prompt)
            history.append({"role": "assistant", "content": prompt})
            print(f"Agata: {prompt}")

            customer_input, classification, ended = wait_for_valid_reply(db, alert_id, history, system_prompt)
            if ended:
                if classification == "NO_ANSWER":
                    print("[No answer on follow-up — sending fallback or finishing call]")
                return

            if classification == "OK":
                classification = "NOT_FRAUD"

            if classification == "FRAUD":
                update_transaction(db, tx.id, is_fraud=True)
                print(f"Transaction {tx.id} marked as FRAUDULENT.")
            elif classification == "NOT_FRAUD":
                update_transaction(db, tx.id, is_fraud=False)
                print(f"Transaction {tx.id} marked as NOT fraudulent.")

            # Generate an agent reply that acknowledges the customer's reply / next steps
            agata_reply = chatgpt_response(history, "Acknowledge the customer's response briefly and state next steps.")
            print(f"Agata: {agata_reply}")
            add_message(db, alert_id, "assistant", agata_reply)
            history.append({"role": "assistant", "content": agata_reply})

        # finished verifying related transactions — wrap up
        wrap_up_call(db, alert_id, history, outcome="FRAUD")
        return

    else:
        # NOT_FRAUD path
        update_transaction(db, alerted_tx.id, is_fraud=False)
        print(f"Transaction {alerted_tx.id} marked as NOT fraudulent.")
        wrap_up_call(db, alert_id, history, outcome="NOT_FRAUD")
        return


if __name__ == "__main__":
    db = next(get_db())
    alert_id = 1

    alert = get_alert(db, alert_id)
    if alert is None:
        print(f"Alert with ID {alert_id} not found.")
        exit(1)

    alerted_tx = db.query(Transaction).filter(Transaction.id == alert.transaction_id).first()
    if alerted_tx is None:
        print(f"Alerted transaction with ID {alert.transaction_id} not found.")
        exit(1)

    recent_txs = db.query(Transaction).filter(Transaction.card_number == alerted_tx.card_number).order_by(Transaction.timestamp.desc()).limit(5).all()

    system_prompt = build_system_prompt(alert, alerted_tx, recent_txs)
    print("\n=== System Prompt ===")
    print(system_prompt)
    print("=====================\n")

    simulate_full_call(db, alert_id, system_prompt, alerted_tx)
