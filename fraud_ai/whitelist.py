from sqlalchemy.orm import Session
from .models import Whitelist
from datetime import datetime, timedelta

# CREATE
def add_to_whitelist(db: Session, card_number: str):
    entry = Whitelist(card_number=card_number)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry

# READ (by card number)
def is_card_whitelisted(db: Session, card_number: str):
    return db.query(Whitelist).filter(Whitelist.card_number == card_number).first()

# READ (all)
def get_whitelist(db: Session):
    return db.query(Whitelist).all()

# DELETE (by card number)
def remove_from_whitelist(db: Session, card_number: str):
    entry = is_card_whitelisted(db, card_number)
    if entry:
        db.delete(entry)
        db.commit()
        return True
    return False

# CLEANUP: Remove expired whitelist entries (older than 30 min)
def cleanup_expired_whitelist(db: Session, expiry_minutes=30):
    expiry_time = datetime.utcnow() - timedelta(minutes=expiry_minutes)
    expired = db.query(Whitelist).filter(Whitelist.whitelisted_at < expiry_time).all()
    for entry in expired:
        db.delete(entry)
    db.commit()
    return len(expired)