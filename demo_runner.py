import logging
from datetime import timedelta
from sqlalchemy.orm import Session
from fraud_ai.data import Base, engine, get_db, create_transaction, Transaction, init_db
from fraud_ai.alerts import create_alert, get_alerts
from fraud_ai.fraud_flow import full_fraud_flow


def get_transactions_last_24h(db: Session, card_number: str, alerted_tx_time, window_hours=24):
    start_time = alerted_tx_time - timedelta(hours=window_hours)
    end_time = alerted_tx_time + timedelta(hours=window_hours)
    return (
        db.query(Transaction)
        .filter(
            Transaction.card_number == card_number,
            Transaction.timestamp >= start_time,
            Transaction.timestamp <= end_time,
        )
        .order_by(Transaction.timestamp.asc())
        .all()
    )


def prepare_demo_db(name="John", surname="Doe"):
    """
    Step 0–1: Reset DB schema, initialize tables, seed demo transactions & alerts.
    Returns (db, alert, last_tx, recent_txs).
    """
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    # Instead of deleting DB file, drop & recreate tables
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    init_db()
    db = next(get_db())

    card_number = "9999888877776666"
    stores = ["Amazon", "Google", "Netflix"]

    last_tx = None
    for i in range(3):
        status = "approved" if i < 2 else "declined"
        tx = create_transaction(
            db,
            card_number=card_number,
            amount=100 + i * 50,
            fraud_score=int(1000 *(0.4 * i)+10),
            is_fraud=False,
            status=status,
            merchant_id=f"M{i+100}",
            merchant_name=stores[i],
            mcc="5999",
            country="US",
            customer_first_name=name,
            customer_last_name=surname,
        )
        last_tx = tx

    alert = None
    if last_tx:
        alert = create_alert(
            db,
            transaction_id=last_tx.id,
            status="open",
            analyst_notes="Auto-generated alert for declined transaction",
        )

    # Get 24h transactions context
    recent_txs = get_transactions_last_24h(db, card_number, last_tx.timestamp)

    return db, alert, last_tx, recent_txs


async def run_fraud_flow_demo(db, alert, last_tx, recent_txs,
                              tts_backend="openai", stt_enabled=False, stt_provider="openai"):
    """
    Step 3: Run the fraud flow — meant to start AFTER frontend loads.
    """
    await full_fraud_flow(
        db=db,
        alert=alert,
        alerted_tx=last_tx,
        recent_txs=recent_txs,
        tts_backend=tts_backend,
        stt_enabled=stt_enabled,
        stt_provider=stt_provider,
    )