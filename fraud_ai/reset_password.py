from sqlalchemy.orm import Session
from fraud_ai.models import PasswordReset
from datetime import datetime

def add_password_reset(db: Session, card_number: str, reason: str = "compromised credentials"):
    reset = PasswordReset(
        card_number=card_number,
        reason=reason,
        timestamp=datetime.utcnow()
    )
    db.add(reset)
    db.commit()
    db.refresh(reset)
    return reset

def has_password_reset(db: Session, card_number: str):
    return db.query(PasswordReset).filter(
        PasswordReset.card_number == card_number
    ).order_by(PasswordReset.timestamp.desc()).first()

def remove_password_reset(db: Session, card_number: str):
    reset = has_password_reset(db, card_number)
    if reset:
        db.delete(reset)
        db.commit()
