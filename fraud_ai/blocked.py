from sqlalchemy.orm import Session
from .models import BlockedCard

# CREATE
def add_to_blocked(db: Session, card_number: str):
    entry = BlockedCard(card_number=card_number)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry

# READ (by card number)
def is_card_blocked(db: Session, card_number: str):
    return db.query(BlockedCard).filter(BlockedCard.card_number == card_number).first()

# READ (all)
def get_blocked_cards(db: Session):
    return db.query(BlockedCard).all()

# DELETE (by card number)
def remove_from_blocked(db: Session, card_number: str):
    entry = is_card_blocked(db, card_number)
    if entry:
        db.delete(entry)
        db.commit()
        return True
    return False