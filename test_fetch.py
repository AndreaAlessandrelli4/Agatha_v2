from fraud_ai.data import get_db
from fraud_ai.alerts import get_alert
from fraud_ai.models import Transaction

def debug_fetch_alert_and_transactions(alert_id, recent_limit=5):
    db = next(get_db())

    alert = get_alert(db, alert_id)
    if not alert:
        print(f"Alert with ID {alert_id} not found.")
        return

    alerted_tx = db.query(Transaction).filter(Transaction.id == alert.transaction_id).first()
    if not alerted_tx:
        print(f"Alerted transaction with ID {alert.transaction_id} not found.")
        return

    print("Alerted Transaction:")
    print(f"  ID: {alerted_tx.id}")
    print(f"  Amount: {alerted_tx.amount}")
    print(f"  Merchant: {alerted_tx.merchant_name}")
    print(f"  Timestamp: {alerted_tx.timestamp}")

    recent_txs = db.query(Transaction).filter(
        Transaction.card_number == alerted_tx.card_number
    ).order_by(Transaction.timestamp.desc()).limit(recent_limit).all()

    print(f"\nLast {recent_limit} transactions for card {alerted_tx.card_number}:")
    for tx in recent_txs:
        print(f"  ID: {tx.id}, Amount: {tx.amount}, Merchant: {tx.merchant_name}, Timestamp: {tx.timestamp}")

if __name__ == "__main__":
    alert_id_to_test = 1  # Change to your alert ID
    debug_fetch_alert_and_transactions(alert_id_to_test)