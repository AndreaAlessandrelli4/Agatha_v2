from fraud_ai.data import get_db, update_transaction

db = next(get_db())

tx_id_to_mark = 1  # Replace with your actual transaction ID

updated_tx = update_transaction(db, tx_id_to_mark, is_fraud=False)
if updated_tx:
    print(f"Transaction {tx_id_to_mark} marked as fraudulent: {updated_tx.is_fraud}")
else:
    print(f"Transaction {tx_id_to_mark} not found.")