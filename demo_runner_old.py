import os
import asyncio
import subprocess
import sys
import logging
from datetime import timedelta
from sqlalchemy.orm import Session
from fraud_ai.data import init_db, get_db, create_transaction, Transaction
from fraud_ai.alerts import create_alert, get_alerts
from fraud_ai.fraud_flow import full_fraud_flow

logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

if os.path.exists("fraud_ai.db"):
    os.remove("fraud_ai.db")

def get_transactions_last_24h(db: Session, card_number: str, alerted_tx_time, window_hours=24):
    start_time = alerted_tx_time - timedelta(hours=window_hours)
    end_time = alerted_tx_time + timedelta(hours=window_hours)
    return db.query(Transaction).filter(
        Transaction.card_number == card_number,
        Transaction.timestamp >= start_time,
        Transaction.timestamp <= end_time
    ).order_by(Transaction.timestamp.asc()).all()

if __name__ == "__main__":
    # === INIT DB + seed ===
    init_db()
    print("Database initialized!")
    db = next(get_db())

    card_number = "9999888877776666"
    stores = ["Amazon", "Google", "Netflix"]
    customer = {"first_name": "John", "last_name": "Doe"}

    for i in range(3):
        status = "approved" if i < 2 else "declined"
        tx = create_transaction(
            db,
            card_number=card_number,
            amount=100 + i * 50,
            fraud_score=1000 * (0.4 * i),
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
        print(f"Inserted transaction {tx.id} for card {card_number} (Status: {status})")

    last_tx = db.query(Transaction).filter(Transaction.id == 2).first()
    if last_tx:
        alert = create_alert(
            db,
            transaction_id=last_tx.id,
            status="open",
            analyst_notes="Auto-generated alert for declined transaction"
        )
        print(f"Created alert {alert.id} for transaction {last_tx.id} (Status: {last_tx.status})")

    print("Current alerts:")
    for a in get_alerts(db):
        print(f"Alert {a.id} for transaction {a.transaction_id}, status: {a.status}")

    recent_txs = get_transactions_last_24h(db, card_number, last_tx.timestamp)


    
    # === THEN run fraud flow in this process (blocking) ===
    tts_backend = "openai"   # or "openai"
    stt_enabled = False
    stt_provider = "openai"

    asyncio.run(full_fraud_flow(
        db,
        alert,
        last_tx,
        recent_txs,
        tts_backend,
        stt_enabled,
        stt_provider,
    ))