from fraud_ai.data import init_db
import os 

try:
    os.remove("fraud_ai.db")
except:
    pass 
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





