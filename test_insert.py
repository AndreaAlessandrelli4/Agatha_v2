from fraud_ai.data import get_db, init_db
from fraud_ai.models import Transaction

init_db()
db = next(get_db())

# Insert a fake transaction with all fields
tx = Transaction(
    card_number="1234567890123456",
    amount=99.99,
    fraud_score=0.85,
    merchant_id="M12345",
    merchant_name="SuperMart",
    mcc="5411",           # Example: 5411 = Grocery Stores
    country="US"
    # timestamp and status will use their default values
)

db.add(tx)
db.commit()
print("Inserted:", tx)