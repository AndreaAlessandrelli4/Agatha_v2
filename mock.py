from fraud_ai.data import get_db, create_transaction
from fraud_ai.alerts import create_alert, get_alerts

db = next(get_db())

card_number = "9999888877776666"  # Same card as your alert

# Insert multiple transactions for the same card and create alerts for them
for i in range(2):
    tx = create_transaction(
        db,
        card_number=card_number,
        amount=100 + i * 50,  # Vary amounts
        fraud_score=0.5 * i,  # Vary fraud scores
        is_fraud=False,
        status="completed" if i % 2 == 0 else "pending",
        merchant_id=f"M{i+100}",
        merchant_name=f"Store {i+1}",
        mcc="5999",
        country="US"
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